"""LangGraph 图节点函数 — Agent 编排的全部逻辑"""

from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .state import AgentGraphState
from .agent_utils import PLANNER_SYSTEM_PROMPT
from .handlers import HANDLER_MAP, handle_fallback

from .cache import agent_cache
from .retry import run_agent_with_retry, simple_quality_score
from src.memory import memory_manager
from src.memory.agent_scoped_view import (
    build_agent_context,
    write_to_context,
    log_creative_decision,
    update_creative_anchors,
    check_anchor_drift_from_review,
    ContentBlock,
    ReviewBlock,
    SEOBlock,
    WriteResult,
)

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from src.observability import get_tracer
from src.observability.metrics import record_agent_task, record_agent_think

logger = logging.getLogger(__name__)

_tracer = None


def _get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = get_tracer("agent.nodes")
    return _tracer

# Token Budget 配置 — 已迁移至 Settings.agent_token_budgets

# 常量/模板已迁移至 agent_utils.py 和各 Agent 模块

# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _resolve_dep_id(dep: str | int, task_queue: list[dict]) -> str | None:
    """将依赖引用统一解析为 task_id（支持数字索引和字符串 ID）

    - 数字索引：取 task_queue[dep] 的 id 字段
    - 字符串 ID：在 task_queue 中查找对应 id 的任务，返回其 id
    - 找不到时返回 None（表示依赖引用无效，应视为未满足）
    """
    if isinstance(dep, int):
        if 0 <= dep < len(task_queue):
            return task_queue[dep].get("id")
        return None
    dep_str = str(dep)
    for t in task_queue:
        if t.get("id") == dep_str:
            return dep_str
    return None

def _classify_complexity(user_request: str) -> str:
    """对用户需求进行复杂度分类"""
    text = user_request.lower()
    complex_keywords = ["图表", "架构图", "流程图", "思维导图", "配图", "封面", "多平台", "报告", "行业分析", "视频", "脚本", "newsletter", "博客", "深度"]
    medium_keywords = ["搜索", "热点", "热搜", "趋势", "优化", "seo", "小红书", "公众号", "知乎", "微博", "推广", "营销"]

    complex_count = sum(1 for kw in complex_keywords if kw in text)
    medium_count = sum(1 for kw in medium_keywords if kw in text)

    if complex_count >= 2 or (complex_count >= 1 and len(text) > 50):
        return "complex"
    elif medium_count >= 1 or len(text) > 80:
        return "medium"
    return "simple"


def _build_template_plan(user_request: str, complexity: str) -> dict:
    """根据复杂度返回预设模板计划"""
    plans = {
        "simple": {
            "goal": user_request[:200],
            "reasoning": "简单任务：直接写作 + 质量审核",
            "estimated_steps": 2,
            "tasks": [
                {"id": str(uuid.uuid4()), "agent_type": "WritingAgent", "input": {"topic": user_request, "content_type": "blog"}, "dependencies": []},
                {"id": str(uuid.uuid4()), "agent_type": "ReviewAgent", "input": {"topic": user_request}, "dependencies": [0]},
            ],
        },
        "medium": {
            "goal": user_request[:200],
            "reasoning": "中等任务：搜索 → 写作 → SEO → 审核",
            "estimated_steps": 4,
            "tasks": [
                {"id": str(uuid.uuid4()), "agent_type": "WebSearch", "input": {"query": user_request, "count": 5}, "dependencies": []},
                {"id": str(uuid.uuid4()), "agent_type": "WritingAgent", "input": {"topic": user_request, "content_type": "blog"}, "dependencies": [0]},
                {"id": str(uuid.uuid4()), "agent_type": "SEOAgent", "input": {"topic": user_request}, "dependencies": [1]},
                {"id": str(uuid.uuid4()), "agent_type": "ReviewAgent", "input": {"topic": user_request}, "dependencies": [1, 2]},
            ],
        },
        "complex": {
            "goal": user_request[:200],
            "reasoning": "复杂任务：搜索 → 写作 → 图表 → SEO → 审核",
            "estimated_steps": 6,
            "tasks": [
                {"id": str(uuid.uuid4()), "agent_type": "WebSearch", "input": {"query": user_request, "count": 8}, "dependencies": []},
                {"id": str(uuid.uuid4()), "agent_type": "WritingAgent", "input": {"topic": user_request, "content_type": "blog"}, "dependencies": [0]},
                {"id": str(uuid.uuid4()), "agent_type": "DiagramAgent", "input": {"topic": user_request, "diagram_type": "auto"}, "dependencies": [1]},
                {"id": str(uuid.uuid4()), "agent_type": "SEOAgent", "input": {"topic": user_request}, "dependencies": [1]},
                {"id": str(uuid.uuid4()), "agent_type": "ReviewAgent", "input": {"topic": user_request}, "dependencies": [1, 2, 3]},
            ],
        },
    }
    return plans.get(complexity, plans["simple"])


def _get_llm(state: AgentGraphState, agent_type: str) -> ChatOpenAI:
    """从 state 中获取对应 agent 类型的 LLM 实例"""
    models = state.get("_llm_models", {})
    llm = models.get(agent_type, models.get("default"))
    if llm is None:
        logger.warning(f"No LLM configured for agent_type=%s, falling back to mock mode", agent_type)
    return llm


def _update_progress(state: AgentGraphState, progress: float, message: str = "", step: str = "") -> None:
    """更新 TaskQueue 中的进度（用于 WebSocket 推送）"""
    task_queue = state.get("_task_queue_ref")
    task_id = state.get("_async_task_id", "")
    if task_queue and task_id:
        task_queue.update_task_progress(task_id, progress, message=message, step=step)


async def _publish_streaming_event(state: AgentGraphState, event_type: str, data: dict) -> None:
    """将事件发布到 EventBus（实时推送给 SSE/WebSocket 前端）"""
    task_queue = state.get("_task_queue_ref")
    task_id = state.get("_async_task_id", "")
    if task_queue and task_id and state.get("_streaming_mode", False):
        try:
            if event_type == "thinking_step":
                await task_queue.publish_thinking_step(task_id, data)
            elif event_type == "llm_chunk":
                await task_queue.publish_llm_chunk(task_id, data)
            elif event_type == "reasoning":
                await task_queue.publish_reasoning(task_id, data)
        except Exception as e:
            logger.warning("Failed to publish streaming event (%s): %s", event_type, e)


def _add_thinking_step(state: AgentGraphState, agent: str, action: str, reasoning: str = "",
                       input_summary: str = "", output_summary: str = "",
                       status: str = "done", tokens: int = 0, elapsed: float = 0) -> dict:
    """添加一个思考步骤到 state"""
    step = {
        "step_number": len(state.get("thinking_steps", [])) + 1,
        "agent": agent,
        "action": action,
        "reasoning": reasoning,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "tokens_used": tokens,
        "time_elapsed": elapsed,
        "status": status,
    }
    return {"thinking_steps": [step]}  # list reducer will append


def _post_process_context(
    agent_type: str,
    result: dict,
    store: Any,
    state: AgentGraphState,
) -> None:
    """
    Post-processing after handler execution:
    1. Log creative decision (WritingAgent only)
    2. Write output to context store via permission system
    3. Extract and update creative anchors (WritingAgent only)
    4. Check anchor drift (ReviewAgent only)
    """
    try:
        if agent_type == "WritingAgent":
            output = result.get("output", {})
            content = output.get("content", "") if isinstance(output, dict) else ""

            # 1. Log creative decision
            decision = content[:200] + ("..." if len(content) > 200 else "")
            log_creative_decision(
                decision_text=decision,
                tags=["writing_output"],
                agent=agent_type,
                store=store,
            )

            # 2. Write content to store via permission system
            if content:
                content_block = ContentBlock(text=content, agent=agent_type)
                write_result = write_to_context(
                    agent_name=agent_type,
                    field_path="task.writing_content",
                    value=content_block,
                    store=store,
                )
                if write_result != WriteResult.ACCEPTED:
                    logger.warning(
                        "WritingAgent content write was not accepted: %s", write_result
                    )

            # 3. Extract and update creative anchors
            updated = update_creative_anchors(result, store)
            if updated:
                logger.info("WritingAgent: updated creative anchors from output")

        elif agent_type == "ReviewAgent":
            # Check anchor drift from review output
            drifts = check_anchor_drift_from_review(result, store)
            if drifts > 0:
                logger.info(
                    "ReviewAgent: detected %d anchor drift signals", drifts
                )

            # Write review feedback to store
            output = result.get("output", {})
            if isinstance(output, dict):
                feedback_text = output.get("content", "")
                passed = output.get("passed", False)
                review_block = ReviewBlock(
                    text=feedback_text,
                    approved=passed,
                )
                write_to_context(
                    agent_name=agent_type,
                    field_path="task.review_feedback",
                    value=review_block,
                    store=store,
                )

    except Exception as e:
        logger.warning("Post-processing context failed for %s: %s", agent_type, e)


def _serialize_context_for_prompt(ctx: dict, agent_type: str) -> str:
    """Serialize a permission-scoped context view into a prompt string."""
    parts = ["[Context]"]

    system = ctx.get("system", {})
    if system.get("project_manifest"):
        pm = system["project_manifest"]
        if hasattr(pm, "title"):
            parts.append(f"Project: {pm.title} | Stage: {pm.stage.value if hasattr(pm.stage, 'value') else pm.stage}")

    if system.get("creative_log"):
        parts.append("\n[Creative Decision History]")
        for d in system["creative_log"][-3:]:
            parts.append(f"  - [{d.agent}] {d.decision_text}")

    task = ctx.get("task", {})
    if task.get("creative_anchors"):
        anchors = task["creative_anchors"]
        if hasattr(anchors, "story_outline") and anchors.story_outline:
            parts.append(f"\n[Story Outline]: {anchors.story_outline}")
        if hasattr(anchors, "character_profiles") and anchors.character_profiles:
            parts.append("[Characters]:")
            for name, profile in anchors.character_profiles.items():
                parts.append(f"  - {name}: {profile}")
        if hasattr(anchors, "world_setting") and anchors.world_setting:
            parts.append(f"\n[World Setting]: {anchors.world_setting}")
        if hasattr(anchors, "plot_milestones") and anchors.plot_milestones:
            parts.append("[Plot Milestones]:")
            for m in anchors.plot_milestones:
                parts.append(f"  - {m}")
        if hasattr(anchors, "style_guide") and anchors.style_guide:
            parts.append(f"\n[Style Guide]: {anchors.style_guide}")

    if agent_type == "WritingAgent" and "creative_view" in ctx:
        cv = ctx["creative_view"]
        if cv.get("active_feedback", {}).get("review"):
            review = cv["active_feedback"]["review"]
            parts.append(f"\n[Review Feedback]: {review.text}")
        if cv.get("recent_content"):
            for content in cv["recent_content"][-2:]:
                text = getattr(content, "text", str(content))
                parts.append(f"\n[Recent Content]: {text[-300:]}")

    parts.append("[/Context]")
    return "\n".join(parts)


# ──────────────────────────────────────────────
# 1. classify_and_decompose 节点
# ──────────────────────────────────────────────

async def classify_and_decompose(state: AgentGraphState) -> dict:
    """复杂度分类 + 任务分解

    复用原有系统的分类器逻辑：
    - 简单任务 → 模板计划（跳过 LLM）
    - 中等/复杂任务 → LLM 拆解 + 模板兜底
    """
    tracer = _get_tracer()
    user_request = state.get("user_request", "")
    trace_id = str(uuid.uuid4())
    complexity = _classify_complexity(user_request)

    with tracer.start_as_current_span(
        "agent.classify",
        kind=trace.SpanKind.INTERNAL,
        attributes={
            "agent.node": "classify_and_decompose",
            "agent.complexity": complexity,
            "agent.user_request.preview": user_request[:200],
            "custom.run_id": state.get("trace_task_id", ""),
        },
    ) as span:
        span.add_event("complexity_classified", {"complexity": complexity})
        existing_plan = state.get("plan")
        if existing_plan and existing_plan.get("tasks"):
            logger.info("Using pre-computed plan from SSE, skipping re-decomposition")
            return {
                "trace_task_id": trace_id,
                "start_time": time.time(),
                "thinking_steps": [{
                    "step_number": 1,
                    "agent": "ManagerAgent",
                    "action": "任务拆解完成",
                    "reasoning": "Using pre-computed plan from SSE endpoint",
                    "input_summary": user_request[:200],
                    "output_summary": f"执行计划：{existing_plan.get('reasoning', '')[:80]}...",
                    "tokens_used": 0,
                    "time_elapsed": 0,
                    "status": "done",
                }],
                "status": "running",
                "task_queue": existing_plan.get("tasks", []),
                "plan_reasoning": existing_plan.get("reasoning", ""),
            }

        updates = {
            "trace_task_id": trace_id,
            "start_time": time.time(),
            "thinking_steps": [],
            "status": "running",
        }

        span.add_event("step_started", {"step": "complexity_classification"})
        updates.update(_add_thinking_step(
            state, "ManagerAgent", "decompose",
            reasoning=f"Analyzing user request: {user_request[:100]}...",
            input_summary=user_request[:200],
        ))

        logger.info(f"Classified request as '{complexity}': {user_request[:80]}")
        span.set_attribute("agent.complexity", complexity)

        if complexity == "simple":
            plan = _build_template_plan(user_request, complexity)
            updates["plan"] = plan
            updates["plan_reasoning"] = plan["reasoning"]
            updates["task_queue"] = plan["tasks"]
            step = {
                "step_number": 1,
                "agent": "ManagerAgent",
                "action": "任务拆解完成",
                "reasoning": f"简单任务，使用模板计划（{len(plan['tasks'])} 步）",
                "input_summary": user_request[:200],
                "output_summary": f"执行计划：{plan['reasoning'][:80]}...",
                "tokens_used": 0,
                "time_elapsed": 0,
                "status": "done",
            }
            updates["thinking_steps"] = [step]
            span.add_event("plan_built", {"task_count": len(plan["tasks"]), "method": "template"})
            span.set_status(Status(StatusCode.OK))
            return updates

        llm = _get_llm(state, "ManagerAgent")
        if not llm:
            plan = _build_template_plan(user_request, complexity)
            updates["plan"] = plan
            updates["plan_reasoning"] = plan["reasoning"]
            updates["task_queue"] = plan["tasks"]
            span.set_status(Status(StatusCode.OK))
            return updates

        span.add_event("llm_decomposition_started")
        try:
            context_str = json.dumps(state.get("context", {}), ensure_ascii=False, indent=2) if state.get("context") else "{}"
            response = await llm.ainvoke([
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=f"User request: {user_request}\n\nComplexity: {complexity}\nContext: {context_str}\n\nBreak this down into sub-tasks."),
            ])
            plan = _parse_plan(response.content, user_request)

            if complexity == "medium" and len(plan.get("tasks", [])) > 8:
                plan = _build_template_plan(user_request, complexity)
            elif complexity == "complex" and len(plan.get("tasks", [])) < 2:
                plan = _build_template_plan(user_request, complexity)

            updates["plan"] = plan
            updates["plan_reasoning"] = plan.get("reasoning", "")
            updates["task_queue"] = plan.get("tasks", [])
            span.add_event("llm_decomposition_completed", {"task_count": len(plan.get("tasks", []))})
        except Exception as e:
            logger.warning(f"LLM decompose failed ({e}), using template")
            plan = _build_template_plan(user_request, complexity)
            updates["plan"] = plan
            updates["plan_reasoning"] = plan["reasoning"]
            updates["task_queue"] = plan["tasks"]
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))

        span.set_attribute("agent.task_count", len(updates.get("task_queue", [])))
        span.set_status(Status(StatusCode.OK))
        return updates


def _parse_plan(response: str, goal: str) -> dict:
    """解析 LLM 返回的计划 JSON"""
    try:
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].strip()
        data = json.loads(json_str)
        tasks = []
        for i, t in enumerate(data.get("tasks", [])):
            tasks.append({
                "id": str(uuid.uuid4()),
                "agent_type": t.get("agent_type", "WritingAgent"),
                "input": t.get("input", {}),
                "dependencies": t.get("dependencies", []),
                "config": {"reasoning": t.get("reasoning", ""), "priority": i},
            })
        return {"goal": data.get("goal", goal), "tasks": tasks, "reasoning": data.get("reasoning", ""), "estimated_steps": len(tasks)}
    except (json.JSONDecodeError, KeyError):
        plan = _build_template_plan(goal, "simple")
        plan["reasoning"] = f"Auto-generated fallback plan (parse error)"
        return plan

# ──────────────────────────────────────────────
# 2. schedule_and_execute 节点（合并原 find_ready_tasks + check_task_results）
# ──────────────────────────────────────────────

async def schedule_and_execute(state: AgentGraphState) -> dict:
    """调度节点：检查 DAG 依赖，找出下一批可执行的任务"""
    tracer = _get_tracer()
    task_queue = state.get("task_queue", [])
    completed_ids = set(state.get("completed_task_ids", []))
    failed_ids = set(state.get("failed_task_ids", []))

    def deps_met(task: dict, task_queue: list[dict], completed_ids: set, failed_ids: set) -> bool:
        for dep in task.get("dependencies", []):
            dep_id = _resolve_dep_id(dep, task_queue)
            if dep_id is None:
                return False
            if dep_id not in completed_ids and dep_id not in failed_ids:
                return False
        return True

    ready = [
        t for t in task_queue
        if t.get("id") not in completed_ids
        and t.get("id") not in failed_ids
        and deps_met(t, task_queue, completed_ids, failed_ids)
    ]

    total_done = len(completed_ids) + len(failed_ids)
    total_tasks = len(task_queue)

    with tracer.start_as_current_span(
        "agent.schedule",
        kind=trace.SpanKind.INTERNAL,
        attributes={
            "agent.node": "schedule_and_execute",
            "schedule.total_tasks": total_tasks,
            "schedule.completed": len(completed_ids),
            "schedule.failed": len(failed_ids),
            "schedule.ready_count": len(ready),
            "custom.run_id": state.get("trace_task_id", ""),
        },
    ) as span:
        if not ready and total_done < total_tasks:
            stuck = [t for t in task_queue if t.get("id") not in completed_ids and t.get("id") not in failed_ids]
            deps_unmet = sum(
                1 for t in stuck for d in t.get("dependencies", [])
                if _resolve_dep_id(d, task_queue) is not None
                and _resolve_dep_id(d, task_queue) not in completed_ids
                and _resolve_dep_id(d, task_queue) not in failed_ids
            )
            if deps_unmet == 0:
                logger.warning(f"Deadlock detected: {len(stuck)} tasks stuck with no unmet dependencies")
                span.add_event("deadlock_detected", {"stuck_count": len(stuck)})
                span.set_status(Status(StatusCode.ERROR, "Deadlock detected"))
                return {
                    "ready_tasks": [],
                    "failed_task_ids": list(state.get("failed_task_ids", [])) + [t["id"] for t in stuck],
                }
            else:
                span.add_event("waiting_for_dependencies", {"deps_unmet": deps_unmet})
                span.set_status(Status(StatusCode.OK))
                return {"ready_tasks": []}

        span.set_status(Status(StatusCode.OK))
        return {"ready_tasks": ready}

# ──────────────────────────────────────────────
# 3. execute_task_batch 节点
# ──────────────────────────────────────────────

async def execute_task_batch(state: AgentGraphState) -> dict:
    """并行执行当前批次的所有任务"""
    tracer = _get_tracer()
    ready_tasks = state.get("ready_tasks", [])
    existing_steps = state.get("thinking_steps", [])
    base_step = len(existing_steps)
    concurrency = state.get("concurrency_limit", 3)
    run_id = state.get("trace_task_id", "")

    with tracer.start_as_current_span(
        "agent.batch_execute",
        kind=trace.SpanKind.INTERNAL,
        attributes={
            "agent.node": "execute_task_batch",
            "batch.size": len(ready_tasks),
            "batch.max_concurrent": concurrency,
            "custom.run_id": run_id,
        },
    ) as batch_span:
        semaphore = asyncio.Semaphore(concurrency)

        async def run_one(task: dict, task_step_number: int) -> tuple[str, dict]:
            task_id = task.get("id", "")
            agent_type = task.get("agent_type", "")

            with tracer.start_as_current_span(
                "agent.task_run",
                kind=trace.SpanKind.INTERNAL,
                attributes={
                    "agent.node": "agent_task_run",
                    "task.id": task_id,
                    "task.type": task.get("type", ""),
                    "task.agent_type": agent_type,
                    "task.priority": task.get("priority", 0),
                    "task.dependencies": json.dumps(task.get("dependencies", [])),
                    "custom.run_id": run_id,
                },
            ) as task_span:
                async with semaphore:
                    try:
                        result = await _run_single_agent_task(task, state, task_step_number=task_step_number)
                        task_span.set_status(Status(StatusCode.OK))
                        task_span.set_attribute("task.status", result.get("status", "unknown"))
                        return task_id, result
                    except Exception as e:
                        task_span.set_status(Status(StatusCode.ERROR, str(e)))
                        task_span.record_exception(e)
                        task_span.set_attribute("task.status", "failed")
                        raise

        tasks_with_steps = [(t, base_step + i + 1) for i, t in enumerate(ready_tasks)]
        batch_span.add_event("batch_started", {"task_count": len(tasks_with_steps)})

        results_list = await asyncio.gather(
            *[run_one(t, sn) for t, sn in tasks_with_steps],
            return_exceptions=True,
        )

        task_results = dict(state.get("task_results", {}))
        completed = list(state.get("completed_task_ids", []))
        failed = list(state.get("failed_task_ids", []))
        new_steps = []

        for task_id, result in results_list:
            if isinstance(result, Exception):
                logger.error("Agent task %s raised exception: %s", task_id, result)
                failed.append(task_id)
                task_results[task_id] = {
                    "status": "failed", "error": str(result), "thinking_steps": [],
                }
                continue
            task_results[task_id] = result
            if result.get("status") in ("completed", "partial"):
                completed.append(task_id)
            else:
                failed.append(task_id)
            if result.get("thinking_steps"):
                new_steps.extend(result["thinking_steps"])

        existing = state.get("thinking_steps", [])
        batch_span.add_event("batch_completed", {
            "completed": len(completed), "failed": len(failed),
        })
        batch_span.set_status(Status(StatusCode.OK))

        record_agent_task(
            agent_type="batch",
            task_type="batch_execute",
            status="completed",
            duration_s=(time.time() - state.get("start_time", time.time())),
        )

        return {
            "task_results": task_results,
            "completed_task_ids": completed,
            "failed_task_ids": failed,
            "thinking_steps": existing + new_steps,
        }
    return {
        "task_results": task_results,
        "completed_task_ids": completed,
        "failed_task_ids": failed,
        "thinking_steps": existing + new_steps,
    }

# ──────────────────────────────────────────────
# 4. merge_and_finish 节点
# ──────────────────────────────────────────────

async def _run_single_agent_task(task: dict, state: AgentGraphState, task_step_number: int = 0) -> dict:
    """根据 agent_type 路由到对应的处理函数

    功能集成：
    - Token Budget 上限控制
    - Agent Cache 缓存复用
    - 上下文隔离（只传 public 内容）
    - 实时流式推送：任务开始、完成、LLM token 均实时推送给前端

    Args:
        task: 待执行的任务 dict
        state: LangGraph 状态（读取用，不直接修改）
        task_step_number: 预先分配的步骤编号（由 execute_task_batch 批量预分配，
                          防止并行执行时竞态导致编号重复）
    """
    agent_type = task.get("agent_type", "")
    llm = _get_llm(state, agent_type)
    task_input = task.get("input", {})
    start_time = time.time()
    task_id = state.get("_async_task_id", "")
    streaming_mode = state.get("_streaming_mode", False)

    # Mock 模式：无 LLM 可用时使用 fallback（有明确的警告日志）
    if not llm:
        logger.warning(
            "Agent %s: no LLM configured (all providers failed). "
            "Returning mock result. Set API keys in .env to enable real generation.",
            agent_type,
        )
        fallback_task = {
            "agent_type": agent_type,
            "input": task_input,
            "_tool_registry": state.get("_tool_registry"),
        }
        return await handle_fallback(fallback_task, llm, None)

    handler = HANDLER_MAP.get(agent_type)
    if handler:
        try:
            t_input = {**task_input}

            # 上下文隔离：只传 public 内容
            dep_context = _build_dependency_context(task, state)
            if dep_context:
                t_input["_context"] = dep_context

            # ── Skill context injection: prepend skill system prompt ──
            skill_system_prompt = state.get("skill_system_prompt", "")
            if skill_system_prompt:
                t_input["_skill_system_prompt"] = skill_system_prompt
                t_input["_skill_name"] = state.get("skill_name", "")

            # MCP 工具传递给支持的 handlers
            mcp_tools = state.get("_mcp_tools", [])
            if mcp_tools:
                t_input["_mcp_tools"] = mcp_tools

            # ── Context management: inject permission-scoped context ──
            context_store = state.get("_context_store")
            if context_store is not None:
                try:
                    ctx = build_agent_context(agent_type, context_store)
                    context_str = _serialize_context_for_prompt(ctx, agent_type)
                    t_input["_extra_context"] = context_str
                except Exception as e:
                    logger.warning("Failed to build agent context: %s", e)

            # 缓存检查：对非写作/分析类 Agent 启用缓存
            cacheable_types = {"SEOAgent", "ReviewAgent", "WebSearch", "HotTopicAgent", "PlatformAdapter"}
            use_cache = agent_type in cacheable_types and not t_input.get("_skip_cache")
            if use_cache:
                cached = agent_cache.get(agent_type, t_input)
                if cached:
                    elapsed = round(time.time() - start_time, 2)
                    cached["time_elapsed"] = elapsed
                    cached["_from_cache"] = True
                    # 实时推送缓存命中（使用预分配的步骤编号）
                    await _publish_streaming_event(state, "thinking_step", {
                        "step_number": task_step_number,
                        "agent": agent_type,
                        "action": task.get("config", {}).get("reasoning", "使用缓存"),
                        "reasoning": f"命中缓存，agent={agent_type}",
                        "input_summary": str(t_input)[:100],
                        "output_summary": cached.get("output", {}).get("content", "")[:200],
                        "tokens_used": cached.get("tokens_used", 0),
                        "time_elapsed": elapsed,
                        "status": "done",
                        "_from_cache": True,
                    })
                    return cached

            # ── 任务开始：发布 thinking_step (running，使用预分配编号) ──
            await _publish_streaming_event(state, "thinking_step", {
                "step_number": task_step_number,
                "agent": agent_type,
                "action": "开始执行",
                "reasoning": f"Agent {agent_type} 开始处理，input={str(t_input)[:150]}",
                "input_summary": str(t_input)[:200],
                "output_summary": "",
                "tokens_used": 0,
                "time_elapsed": 0,
                "status": "running",
            })

            # Token Budget 检查
            from src.config import get_settings
            settings = get_settings()
            budget = settings.agent_token_budgets.get(agent_type, 2000)
            if budget > 0:
                max_tokens = getattr(llm, 'max_tokens', None) or 0
                if max_tokens and max_tokens > budget:
                    logger.warning(
                        "Token budget exceeded for %s: max_tokens=%d > budget=%d",
                        agent_type, max_tokens, budget,
                    )

            # ── 执行 handler（支持流式 LLM 回调）──
            if streaming_mode and agent_type in {"WritingAgent", "SEOAgent", "ReviewAgent"}:
                # 流式 LLM token 推送用 fire-and-forget：LLM 流是主干，不应被推送阻塞
                def _make_on_llm_chunk(agent: str):
                    async def _on_llm_chunk(chunk: str) -> None:
                        asyncio.create_task(_publish_streaming_event(
                            state, "llm_chunk",
                            {"agent": agent, "content": chunk, "timestamp": time.time()}
                        ))
                    return _on_llm_chunk
                def _make_on_reasoning(agent: str):
                    async def _on_reasoning(text: str) -> None:
                        asyncio.create_task(_publish_streaming_event(
                            state, "reasoning",
                            {"agent": agent, "content": text, "timestamp": time.time()}
                        ))
                    return _on_reasoning
                t_input["_on_llm_chunk"] = _make_on_llm_chunk(agent_type)
                t_input["_on_reasoning"] = _make_on_reasoning(agent_type)

            # ── 记忆注入：从语义/情景/工作记忆构建增强上下文 ──
            try:
                memory_context = await memory_manager.build_agent_context(
                    agent_type=agent_type,
                    task_input=t_input,
                    user_id=state.get("context", {}).get("user_id", ""),
                )
                if memory_context:
                    t_input["_memory_context"] = memory_context
            except Exception as e:
                logger.warning(f"Memory context injection failed: {e}")

            # ── 消息总线注入（Phase 2 点对点通信）──
            if state.get("use_message_bus") and state.get("_message_bus"):
                t_input["_message_bus"] = state["_message_bus"]
                # 同时注入 AgentRegistry 供 handler 按名称查找 Agent
                t_input["_agent_registry"] = state.get("_agent_registry")

            # ── ToolRegistry 注入（Tool System）──
            tool_registry = state.get("_tool_registry")
            if tool_registry is not None:
                t_input["_tool_registry"] = tool_registry
                # 为启用了工具的 Agent 注入工具描述
                from .agents import PREDEFINED_PROFILES
                profile = PREDEFINED_PROFILES.get(agent_type)
                if profile and profile.tools_enabled:
                    from .tools import tool_prompt_section
                    tool_desc = tool_prompt_section(agent_type, tool_registry)
                    if tool_desc:
                        t_input["_tool_descriptions"] = tool_desc

            # 使用 retry wrapper 执行 handler（对产出型 Agent 启用质量检查与重试）
            # 注意：这里有两个质量保障机制：
            #   1. run_agent_with_retry — 基于浅层指标（长度/结构/占位符），快速反馈
            #   2. reflect_on_task_batch — 基于 LLM 自省，深度评估
            # 两者不会同时生效：reflection 启用时跳过 retry 的质量重试，
            # 否则同一个任务可能被两套机制各重试一次。
            retry_types = {"WritingAgent", "ReviewAgent", "CodeAgent", "TranslationAgent", "SEOAgent", "DiagramAgent"}
            _mcp = mcp_tools if mcp_tools else t_input.get("_mcp_tools")
            _handler = handler
            if agent_type in retry_types:
                if state.get("enable_reflection"):
                    # Reflection 模式下跳过 retry 的质量评分，直接执行
                    # 质量判断由 reflect_on_task_batch 统一负责
                    result = await _handler(t_input, llm, _mcp)
                else:
                    # 无 Reflection 时，retry 负责浅层质量检查和重试
                    _adapter = lambda ti, llm, _st: _handler(ti, llm, _mcp)
                    result = await run_agent_with_retry(
                        agent_func=_adapter,
                        task_input=t_input,
                        llm=llm,
                        state=state,
                        max_retries=2,
                        agent_type=agent_type,
                    )
            else:
                result = await _handler(t_input, llm, _mcp)

            # ── Context management: post-processing after handler ──
            context_store = state.get("_context_store")
            if context_store is not None and result.get("status") in ("completed", "partial"):
                _post_process_context(agent_type, result, context_store, state)

            # 清理回调引用（不可 JSON 序列化），在后续操作前移除
            t_input.pop("_on_llm_chunk", None)
            t_input.pop("_on_reasoning", None)

            elapsed = round(time.time() - start_time, 2)
            result["time_elapsed"] = elapsed
            if "thinking_steps" not in result:
                result["thinking_steps"] = []

            # ── 任务完成：发布 thinking_step (done) ──
            output_content = result.get("output", {}).get("content", "")
            await _publish_streaming_event(state, "thinking_step", {
                "step_number": task_step_number,
                "agent": agent_type,
                "action": "完成",
                "reasoning": f"Agent {agent_type} 执行完成",
                "input_summary": str(t_input)[:200],
                "output_summary": output_content[:300] if output_content else (result.get("error", "") or "完成"),
                "tokens_used": result.get("tokens_used", 0),
                "time_elapsed": elapsed,
                "status": result.get("status", "completed"),
            })

            # 缓存结果（清理内部字段防止序列化失败）
            if use_cache and result.get("status") in ("completed", "partial"):
                try:
                    _cache_input = {k: v for k, v in t_input.items() if not k.startswith('_')}
                    agent_cache.set(agent_type, _cache_input, result)
                except Exception as e:
                    logger.warning(f"Failed to cache result for {agent_type}: {e}")

            return result
        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            logger.error(f"Agent {agent_type} failed: {e}")
            # ── 任务失败：发布 thinking_step (failed) ──
            await _publish_streaming_event(state, "thinking_step", {
                "step_number": task_step_number,
                "agent": agent_type,
                "action": "失败",
                "reasoning": str(e),
                "input_summary": str(task_input)[:200],
                "output_summary": str(e),
                "tokens_used": 0,
                "time_elapsed": elapsed,
                "status": "failed",
            })
            return {
                "status": "failed",
                "agent_type": agent_type,
                "error": str(e),
                "time_elapsed": elapsed,
                "thinking_steps": [{"agent": agent_type, "action": "error", "reasoning": str(e), "status": "failed"}],
            }

    return await handle_fallback({"agent_type": agent_type, "input": task_input, "_tool_registry": state.get("_tool_registry")}, llm, None)


def _build_dependency_context(task: dict, state: AgentGraphState) -> dict:
    """构建依赖上下文（上下文隔离：只传递 content 字段，防止 Context Bleeding）

    对应 AGENT_ARCHITECTURE.md §14.1：
    - 不传递生 Agent 之间的内部指令、system prompt、中间推理过程
    - 只保留用户可见的内容字段给下游
    """
    task_queue = state.get("task_queue", [])
    task_results = state.get("task_results", {})
    context = {}

    for dep in task.get("dependencies", []):
        dep_id = _resolve_dep_id(dep, task_queue)
        # 跳过无效的依赖引用（索引越界或 ID 不存在）
        if dep_id is None:
            continue
        if dep_id in task_results:
            prev = task_results[dep_id]
            output = prev.get("output", {})
            # output 可能是字符串（fallback handler）也可能是 dict，统一提取 content
            if isinstance(output, dict):
                content = output.get("content", "")
            elif isinstance(output, str):
                content = output
            else:
                content = str(output) if output else ""
            context[dep_id] = {
                "content": content,
                "agent": prev.get("agent_type", ""),
            }
    return context if context else {}


# ──────────────────────────────────────────────
# 5. merge_and_finish 节点
# ──────────────────────────────────────────────

async def merge_and_finish(state: AgentGraphState) -> dict:
    """合并所有子任务结果为最终输出 + 持久化"""
    tracer = _get_tracer()
    plan = state.get("plan") or {}
    task_results = state.get("task_results", {})
    task_queue = state.get("task_queue", [])
    completed_ids = set(state.get("completed_task_ids", []))
    failed_ids = set(state.get("failed_task_ids", []))

    with tracer.start_as_current_span(
        "agent.merge",
        kind=trace.SpanKind.INTERNAL,
        attributes={
            "agent.node": "merge_and_finish",
            "merge.total_tasks": len(task_queue),
            "merge.completed": len(completed_ids),
            "merge.failed": len(failed_ids),
            "custom.run_id": state.get("trace_task_id", ""),
        },
    ) as merge_span:
        sections = []
        full_text_parts = []

        for task in task_queue:
            tid = task.get("id", "")
            result = task_results.get(tid, {})
            if tid in completed_ids or tid in failed_ids:
                sections.append({
                    "agent": result.get("agent_type", task.get("agent_type", "")),
                    "task_input": task.get("input", {}),
                    "output": result.get("output", {}),
                    "status": result.get("status", "unknown"),
                    "error": result.get("error"),
                })
                content = result.get("output", {}).get("content", "")
                if content:
                    full_text_parts.append(f"\n\n--- {result.get('agent_type', task.get('agent_type', ''))} ---\n{content}")

                try:
                    await memory_manager.remember(
                        agent_type=result.get("agent_type", task.get("agent_type", "")),
                        task_input=task.get("input", {}),
                        result=result,
                        user_id=state.get("context", {}).get("user_id", ""),
                    )
                except Exception as e:
                    logger.warning("Memory remember failed for %s: %s",
                                   result.get("agent_type", "unknown"), e)

        completed = len(completed_ids)
        failed = len(failed_ids)

        status = "completed"
        if failed > 0 and completed == 0:
            status = "failed"
        elif failed > 0:
            status = "partial"

        asyncio.create_task(_persist_results_async(state, sections, full_text_parts, status))

        enable_ctx = state.get("enable_context_management", False)
        context_store = state.get("_context_store")
        if enable_ctx and context_store is not None:
            try:
                from src.memory.consolidator import ConsolidatorNode
                consolidator = ConsolidatorNode()
                result = consolidator.finalize(context_store)
                logger.info(
                    "ConsolidatorNode finalized: status=%s, drifts=%d, entities=%d",
                    result.status, len(result.anchor_drifts), len(result.new_memory_entries),
                )
                merge_span.add_event("consolidator_finalized", {
                    "drifts": len(result.anchor_drifts),
                    "memory_entries": len(result.new_memory_entries),
                })
            except Exception as e:
                logger.warning("ConsolidatorNode.finalize failed: %s", e)

        merge_span.set_attribute("merge.status", status)
        merge_span.set_status(Status(StatusCode.OK))

        return {
            "final_output": {
                "goal": plan.get("goal", ""),
                "plan_reasoning": plan.get("reasoning", ""),
                "sections": sections,
                "full_text": "".join(full_text_parts),
            },
            "status": status,
        }


def _persist_results_sync(state: AgentGraphState, sections: list, full_text_parts: list, status: str) -> None:
    """同步持久化执行结果到数据库"""
    try:
        from .persistence import save_agent_task, save_subtask

        task_id = state.get("_async_task_id", "")
        if not task_id:
            return

        # 任务级持久化
        save_agent_task(
            task_id=task_id or state.get("trace_task_id", ""),
            user_id=state.get("context", {}).get("user_id", ""),
            prompt=state.get("user_request", ""),
            status=status,
            result={"sections": sections, "full_text": "".join(full_text_parts)},
            total_steps=len(state.get("task_queue", [])),
            completed_steps=len(state.get("completed_task_ids", [])),
            total_tokens=sum(s.get("tokens_used", 0) for s in state.get("task_results", {}).values()),
            thinking_trace={"steps": state.get("thinking_steps", [])},
        )

        # 子任务级持久化
        for task in state.get("task_queue", []):
            tid = task.get("id", "")
            result = state.get("task_results", {}).get(tid, {})
            if result:
                save_subtask(
                    task_id=task_id,
                    agent_type=result.get("agent_type", task.get("agent_type", "")),
                    status=result.get("status", "unknown"),
                    input_data=task.get("input", {}),
                    output_data=result.get("output", {}),
                    tokens_used=result.get("tokens_used", 0),
                    time_elapsed=result.get("time_elapsed", 0.0),
                    error=result.get("error"),
                )
    except Exception as e:
        logger.warning(f"Persistence failed: {e}")


async def _persist_results_async(state: AgentGraphState, sections: list, full_text_parts: list, status: str) -> None:
    """异步持久化执行结果到数据库（避免阻塞 LangGraph 流程）"""
    try:
        await asyncio.to_thread(_persist_results_sync, state, sections, full_text_parts, status)
    except Exception as e:
        logger.warning(f"Async persistence failed: {e}")


# ──────────────────────────────────────────────
# 路由函数（用于条件边）
# ──────────────────────────────────────────────

def should_continue_or_finish(state: AgentGraphState) -> str:
    """条件边路由函数：判断是否继续执行任务

    返回值：
    - "execute": 有可执行的任务，继续执行
    - "done": 所有任务已完成，或无可执行任务且无死锁

    逻辑：
    1. 如果 ready_tasks 有任务 → "execute"
    2. 如果没有 ready_tasks 且还有未完成任务 → "done"（死锁或依赖阻塞）
    3. 所有任务都已完成或失败 → "done"
    """
    task_queue = state.get("task_queue", [])
    completed = len(state.get("completed_task_ids", []))
    failed = len(state.get("failed_task_ids", []))
    ready = state.get("ready_tasks", [])
    total_tasks = len(task_queue)
    total_done = completed + failed

    # 情况1：有可执行的任务
    if ready:
        return "execute"

    # 情况2：所有任务已完成或失败
    if total_done >= total_tasks:
        return "done"

    # 情况3：没有可执行的任务但还有未完成的任务（死锁）
    if total_done < total_tasks:
        logger.warning(
            f"No ready tasks ({total_tasks} total, "
            f"{total_done} done/failed). Forcing exit."
        )
        return "done"

    return "done"


# ──────────────────────────────────────────────
# 6. reflect_on_task_batch 节点（Phase 3 反思）
# ──────────────────────────────────────────────

async def reflect_on_task_batch(state: AgentGraphState) -> dict:
    """反思：对已完成批次进行自省和同行评审"""
    if not state.get("enable_reflection"):
        return {}

    tracer = _get_tracer()
    task_results = state.get("task_results", {})
    completed_ids = set(state.get("completed_task_ids", []))
    reflection_results = list(state.get("reflection_results", []))
    reflection_rounds = dict(state.get("self_reflection_rounds", {}))
    llm_models = state.get("_llm_models", {})
    new_steps = list(state.get("thinking_steps", []))

    with tracer.start_as_current_span(
        "agent.reflect",
        kind=trace.SpanKind.INTERNAL,
        attributes={
            "agent.node": "reflect_on_task_batch",
            "reflect.task_count": len(completed_ids),
            "custom.run_id": state.get("trace_task_id", ""),
        },
    ) as reflect_span:
        requeue_ids = set()
        for task_id in completed_ids:
            result = task_results.get(task_id, {})
            agent_type = result.get("agent_type", "")

            if agent_type not in {"WritingAgent", "CodeAgent", "TranslationAgent"}:
                continue

            current_round = reflection_rounds.get(task_id, 0)
            if current_round >= 2:
                continue

            from .reflection import self_reflect as do_self_reflect
            llm = llm_models.get(agent_type) or llm_models.get("default")
            task_input = {"_task_id": task_id}

            with tracer.start_as_current_span(
                "agent.reflect_single",
                kind=trace.SpanKind.INTERNAL,
                attributes={
                    "task.id": task_id,
                    "task.agent_type": agent_type,
                    "reflection.round": current_round + 1,
                },
            ) as single_span:
                try:
                    ref_result = await do_self_reflect(agent_type, task_input, result, llm)
                    if ref_result:
                        ref_dict = ref_result.to_dict()
                        ref_dict["round"] = current_round + 1
                        reflection_results.append(ref_dict)
                        reflection_rounds[task_id] = current_round + 1
                        single_span.set_attribute("reflection.score", ref_result.score)
                        single_span.set_attribute("reflection.passed", ref_result.passed)

                        new_steps.append({
                            "step_number": len(new_steps) + 1,
                            "agent": agent_type,
                            "action": "自省完成",
                            "reasoning": f"自省评分 {ref_result.score}，{len(ref_result.issues)} 个问题",
                            "output_summary": "; ".join(ref_result.suggestions[:3]) if ref_result.suggestions else "无需修改",
                            "tokens_used": 0,
                            "status": "done",
                        })

                        if not ref_result.passed and current_round < 2:
                            requeue_ids.add(task_id)
                            reflect_span.add_event("retry_queued", {"task_id": task_id, "score": ref_result.score})
                except Exception as e:
                    logger.warning("Reflection failed for %s: %s", agent_type, e)
                    single_span.record_exception(e)
                    single_span.set_status(Status(StatusCode.ERROR, str(e)))

        reflect_span.add_event("reflection_completed", {
            "retried": len(requeue_ids),
            "total_reflected": len(completed_ids),
        })

        updated_completed = [tid for tid in completed_ids if tid not in requeue_ids]
        updated_task_results = {tid: res for tid, res in task_results.items() if tid not in requeue_ids}
        reflect_span.set_status(Status(StatusCode.OK))

        return {
            "reflection_results": reflection_results,
            "self_reflection_rounds": reflection_rounds,
            "thinking_steps": new_steps,
            "completed_task_ids": updated_completed,
            "task_results": updated_task_results,
        }


# ──────────────────────────────────────────────
# 7. check_hitl_gates 节点（Phase 4 人机协同）
# ──────────────────────────────────────────────

async def check_hitl_gates(state: AgentGraphState) -> dict:
    """检查 HITL 门状态（观测节点）

    注意：此节点不阻塞执行。实际的 HITL 阻塞由 handler 内部通过
    TaskQueue.set_decision_required / orchestrator.request_user_decision 处理。
    此节点仅用于状态观测和日志记录，为未来 Graph 级 HITL 暂停预留接口。
    """
    if not state.get("enable_hitl"):
        return {}

    pending_gates = state.get("pending_gates", [])
    if pending_gates:
        hitl_mgr = state.get("_hitl_manager")
        unresolved = []
        if hitl_mgr:
            for gate in pending_gates:
                g = hitl_mgr.get_gate(gate.get("gate_id", ""))
                if g and getattr(g, "state", "") != "resolved":
                    unresolved.append(gate)
        if unresolved:
            logger.info(
                "HITL: %d pending gates (blocking handled at TaskQueue level)",
                len(unresolved),
            )
            return {"pending_gates": unresolved}

    return {"pending_gates": []}
