"""LangGraph 图节点函数 — Agent 编排的全部逻辑（简化版）

删除了以下功能：
- SubAgent ReAct 循环（react/）
- MessageBus 对等通信（communication/）
- 反思回路（reflection/）
- HITL 门控（hitl/）
- ToolRegistry 工具注册中心（tools/）
"""

from __future__ import annotations
import asyncio
import json
import time
import uuid
from typing import Any

import structlog
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .state import AgentGraphState
from .agent_utils import PLANNER_SYSTEM_PROMPT, extract_plan_from_llm_response, validate_plan_structure
from .platform_registry import PLATFORMS, detect_platform, get_names_map
from .handlers import HANDLER_MAP, handle_fallback
from .cache import agent_cache

from src.core.exceptions import HandlerError, ToolExecutionError
from src.observability.metrics import record_agent_task, record_agent_run
from src.observability.agent_tracer import get_agent_tracer, AgentTracer

logger = structlog.get_logger(__name__)

_tracer: AgentTracer | None = None


def _get_tracer() -> AgentTracer:
    global _tracer
    if _tracer is None:
        _tracer = get_agent_tracer()
    return _tracer


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _resolve_dep_id(dep: str | int, task_queue: list[dict]) -> str | None:
    """将依赖引用统一解析为 task_id（支持数字索引和字符串 ID）"""
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
    medium_keywords = ["搜索", "热点", "热搜", "趋势", "优化", "seo", "推广", "营销"]

    complex_count = sum(1 for kw in complex_keywords if kw in text)
    medium_count = sum(1 for kw in medium_keywords if kw in text)

    if complex_count >= 2 or (complex_count >= 1 and len(text) > 50):
        return "complex"
    elif medium_count >= 1 or len(text) > 80:
        return "medium"
    return "simple"


def _build_fallback_plan(user_request: str, platform_code: str | None = None) -> dict:
    """极简兜底：只用 WritingAgent + ReviewAgent"""
    goal = user_request[:200]
    detected = platform_code or detect_platform(user_request)
    p = PLATFORMS.get(detected) if detected else None
    content_type = p.content_type if p else "blog"

    task0 = {
        "id": str(uuid.uuid4()), "agent_type": "WritingAgent",
        "input": {"topic": user_request, "content_type": content_type, "platform": detected or ""},
        "dependencies": [], "can_parallel_with": [], "retry_on_low_quality": True, "priority": "high",
    }
    task1 = {
        "id": str(uuid.uuid4()), "agent_type": "ReviewAgent",
        "input": {"topic": user_request, "content_type": content_type, "platform": detected or ""},
        "dependencies": [0], "can_parallel_with": [], "retry_on_low_quality": False, "priority": "medium",
    }
    platform_hint = f"（{p.name}）" if p else ""
    return {
        "goal": goal,
        "tasks": [task0, task1],
        "reasoning": f"兜底计划{platform_hint}：写作 + 审核",
        "plan_method": "template",
        "platform": detected,
    }


def _get_llm(state: AgentGraphState, agent_type: str) -> ChatOpenAI:
    """从 state 中获取对应 agent 类型的 LLM 实例"""
    models = state.get("_llm_models", {})
    llm = models.get(agent_type, models.get("default"))
    if llm is None:
        logger.warning("未配置 LLM，回退到模拟模式", agent_type=agent_type)
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
            logger.warning("发布流式事件失败", event_type=event_type, error=str(e))


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
    return {"thinking_steps": [step]}


def _serialize_context_for_prompt(ctx: dict, agent_type: str) -> str:
    """将上下文视图序列化为提示字符串。"""
    parts = ["[Context]"]
    system = ctx.get("system", {})
    if system.get("project_manifest"):
        pm = system["project_manifest"]
        if hasattr(pm, "title"):
            parts.append(f"Project: {pm.title}")

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


def _post_process_context(agent_type: str, result: dict, store: Any, state: AgentGraphState) -> None:
    """Handler 执行后的后处理"""
    from src.memory.agent_scoped_view import (
        write_to_context, log_creative_decision, update_creative_anchors,
        check_anchor_drift_from_review, ContentBlock, ReviewBlock, WriteResult,
    )

    try:
        if agent_type == "WritingAgent":
            output = result.get("output", {})
            content = output.get("content", "") if isinstance(output, dict) else ""

            decision = content[:200] + ("..." if len(content) > 200 else "")
            log_creative_decision(decision_text=decision, tags=["writing_output"], agent=agent_type, store=store)

            if content:
                content_block = ContentBlock(text=content, agent=agent_type)
                write_result = write_to_context(agent_name=agent_type, field_path="task.writing_content", value=content_block, store=store)
                if write_result != WriteResult.ACCEPTED:
                    logger.warning("WritingAgent 内容写入未被接受", write_result=write_result)

            updated = update_creative_anchors(result, store)
            if updated:
                logger.info("WritingAgent 从输出中更新了创意锚点")

        elif agent_type == "ReviewAgent":
            drifts = check_anchor_drift_from_review(result, store)
            if drifts > 0:
                logger.info("ReviewAgent 检测到锚点偏移信号", drift_count=drifts)

            output = result.get("output", {})
            if isinstance(output, dict):
                feedback_text = output.get("content", "")
                passed = output.get("passed", False)
                review_block = ReviewBlock(text=feedback_text, approved=passed)
                write_to_context(agent_name=agent_type, field_path="task.review_feedback", value=review_block, store=store)
    except Exception as e:
        logger.warning("后处理上下文失败", agent_type=agent_type, error=str(e))


# ──────────────────────────────────────────────
# 1. classify_and_decompose 节点
# ──────────────────────────────────────────────

async def classify_and_decompose(state: AgentGraphState) -> dict:
    """复杂度分类 + 任务分解。LLM 是主路径，模板是紧急兜底。"""
    tracer = _get_tracer()
    user_request = state.get("user_request", "")
    trace_id = str(uuid.uuid4())
    complexity = _classify_complexity(user_request)

    with tracer.start_as_current_span(
        "agent.classify",
        kind="INTERNAL",
        attributes={
            "agent.node": "classify_and_decompose",
            "agent.complexity": complexity,
            "agent.user_request.preview": user_request[:200],
            "custom.run_id": state.get("trace_task_id", ""),
        },
    ) as span:
        span.add_event("complexity_classified", {"complexity": complexity})

        # 如果上游已经提供了计划（来自 SSE 端点），直接复用
        existing_plan = state.get("plan")
        if existing_plan and existing_plan.get("tasks"):
            logger.info("使用 SSE 预计算的计划，跳过重新分解")
            return {
                "trace_task_id": trace_id,
                "start_time": time.time(),
                "thinking_steps": [{
                    "step_number": 1,
                    "agent": "ManagerAgent",
                    "action": "任务拆解完成",
                    "reasoning": "使用 SSE 端点预计算的计划",
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
            reasoning=f"正在分析用户请求：{user_request[:100]}...",
            input_summary=user_request[:200],
        ))

        logger.info("已分类请求复杂度", complexity=complexity, preview=user_request[:80])
        span.set_attribute("agent.complexity", complexity)

        detected = detect_platform(user_request)
        names_map = get_names_map()
        platform = detected if detected else "generic"
        platform_name = names_map.get(platform, "通用内容")
        span.set_attribute("agent.platform", platform)
        logger.info("检测到平台：%s，请求：%s", platform_name, user_request[:60])

        # 主路径：尝试 LLM 分解
        llm = _get_llm(state, "ManagerAgent")
        if llm:
            try:
                context_str = json.dumps(state.get("context", {}), ensure_ascii=False, indent=2) if state.get("context") else "{}"
                response = await llm.ainvoke([
                    SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                    HumanMessage(content=f"用户请求：{user_request}\n\n复杂度：{complexity}\n检测到的平台：{platform_name}\n上下文：{context_str}\n\n请将其分解为子任务，根据检测到的平台进行调整。"),
                ])
                plan = _parse_plan(response.content, user_request, platform=detected)

                if len(plan.get("tasks", [])) == 0:
                    logger.warning("LLM 计划没有任务，使用兜底方案")
                    plan = _build_fallback_plan(user_request)
                else:
                    plan["plan_method"] = "llm"
                    logger.info("LLM 计划已生成：%d 个任务", len(plan.get("tasks", [])))

                updates["plan"] = plan
                updates["plan_reasoning"] = plan.get("reasoning", "")
                updates["task_queue"] = plan.get("tasks", [])
                updates["thinking_steps"].append({
                    "step_number": len(updates["thinking_steps"]) + 1,
                    "agent": "ManagerAgent",
                    "action": "任务拆解完成",
                    "reasoning": f"{plan.get('plan_method', 'llm')} 规划完成，共 {len(plan.get('tasks', []))} 个任务（平台={platform_name}）",
                    "input_summary": user_request[:200],
                    "output_summary": f"执行计划：{plan.get('reasoning', '')[:80]}...",
                    "tokens_used": 0,
                    "time_elapsed": 0,
                    "status": "done",
                })
                span.add_event("plan_built", {"task_count": len(plan.get("tasks", [])), "method": plan.get("plan_method", "llm")})
                span.set_attribute("plan.method", plan.get("plan_method", "llm"))
                span.set_attribute("agent.task_count", len(plan.get("tasks", [])))
                span.set_status("OK")
                return updates

            except Exception as e:
                logger.warning("LLM 分解失败，使用模板兜底", error=str(e))
                span.record_exception(e)

        # 兜底路径：LLM 不可用或分解失败时使用模板
        plan = _build_fallback_plan(user_request)
        updates["plan"] = plan
        updates["plan_reasoning"] = plan["reasoning"]
        updates["task_queue"] = plan["tasks"]
        updates["thinking_steps"].append({
            "step_number": len(updates["thinking_steps"]) + 1,
            "agent": "ManagerAgent",
            "action": "任务拆解完成（LLM兜底）",
            "reasoning": f"平台={platform_name}，{len(plan['tasks'])} 个任务",
            "input_summary": user_request[:200],
            "output_summary": f"执行计划：{plan['reasoning'][:80]}...",
            "tokens_used": 0,
            "time_elapsed": 0,
            "status": "done",
        })
        span.add_event("plan_built", {"task_count": len(plan["tasks"]), "method": "template", "platform": detected})
        span.set_attribute("plan.method", "template")
        span.set_attribute("agent.task_count", len(plan["tasks"]))
        span.set_attribute("agent.platform", detected or "none")
        span.set_status("OK")
        return updates


def _parse_plan(response: str, goal: str, platform: str | None = None) -> dict:
    """解析 LLM 返回的计划 JSON"""
    try:
        plan_dict = extract_plan_from_llm_response(response)
        is_valid, error_msg = validate_plan_structure(plan_dict)
        if not is_valid:
            raise ValueError(f"无效的计划结构：{error_msg}")

        tasks = []
        for i, t in enumerate(plan_dict.get("tasks", [])):
            tasks.append({
                "id": str(uuid.uuid4()),
                "agent_type": t.get("agent_type", "WritingAgent"),
                "input": t.get("input", {}),
                "dependencies": t.get("dependencies", []),
                "can_parallel_with": t.get("can_parallel_with", []),
                "retry_on_low_quality": t.get("retry_on_low_quality", False),
                "priority": t.get("priority", "medium"),
                "config": {"reasoning": t.get("reasoning", ""), "priority": i},
            })
        result = {
            "goal": plan_dict.get("goal", goal),
            "tasks": tasks,
            "reasoning": plan_dict.get("reasoning", ""),
            "estimated_steps": len(tasks),
            "plan_method": "llm",
        }
        if platform:
            result["platform"] = platform
        return result
    except Exception:
        return _build_fallback_plan(goal)


# ──────────────────────────────────────────────
# 2. schedule_and_execute 节点
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
        kind="INTERNAL",
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
                logger.warning("检测到死锁", stuck_count=len(stuck))
                span.add_event("deadlock_detected", {"stuck_count": len(stuck)})
                span.set_status("ERROR")
                return {
                    "ready_tasks": [],
                    "failed_task_ids": list(state.get("failed_task_ids", [])) + [t["id"] for t in stuck],
                }
            else:
                span.add_event("waiting_for_dependencies", {"deps_unmet": deps_unmet})
                span.set_status("OK")
                return {"ready_tasks": []}

        span.set_status("OK")
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
        kind="INTERNAL",
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
                kind="INTERNAL",
                attributes={
                    "agent.node": "agent_task_run",
                    "task.id": task_id,
                    "task.agent_type": agent_type,
                    "task.priority": task.get("priority", 0),
                    "custom.run_id": run_id,
                },
            ) as task_span:
                async with semaphore:
                    try:
                        result = await _run_single_agent_task(task, state, task_step_number=task_step_number)
                        task_span.set_status("OK")
                        task_span.set_attribute("task.status", result.get("status", "unknown"))
                        return task_id, result
                    except Exception as e:
                        task_span.set_status("ERROR")
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

        for i, entry in enumerate(results_list):
            if isinstance(entry, BaseException):
                task = tasks_with_steps[i][0] if i < len(tasks_with_steps) else {"id": f"unknown_{i}"}
                task_id = task.get("id", f"unknown_{i}")
                logger.error("Agent 任务抛出异常", task_id=task_id, exc_info=True, error=str(entry))
                failed.append(task_id)
                task_results[task_id] = {
                    "status": "failed", "error": str(entry), "thinking_steps": [],
                }
                continue
            task_id, result = entry
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
        batch_span.set_status("OK")

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


# ──────────────────────────────────────────────
# 4. _run_single_agent_task（核心执行）
# ──────────────────────────────────────────────

async def _run_single_agent_task(task: dict, state: AgentGraphState, task_step_number: int = 0) -> dict:
    """根据 agent_type 路由到对应的 handler

    Args:
        task: 待执行的任务 dict
        state: LangGraph 状态
        task_step_number: 预先分配的步骤编号
    """
    agent_type = task.get("agent_type", "")
    llm = _get_llm(state, agent_type)
    task_input = task.get("input", {})
    start_time = time.time()
    task_id = state.get("_async_task_id", "")
    streaming_mode = state.get("_streaming_mode", False)

    # Mock 模式：无 LLM 可用时使用 fallback
    if not llm:
        logger.warning("Agent 未配置 LLM，返回模拟结果", agent_type=agent_type)
        fallback_task = {
            "agent_type": agent_type,
            "input": task_input,
        }
        return await handle_fallback(fallback_task, llm, None)

    handler = HANDLER_MAP.get(agent_type)
    if handler:
        try:
            t_input = {**task_input}
            task_session_id = (
                state.get("context", {}).get("session_id")
                or state.get("context", {}).get("conversation_id")
                or "default"
            )
            t_input["_task_id"] = task.get("id") or task_id or agent_type
            t_input["_session_id"] = task_session_id
            t_input["session_id"] = task_session_id

            # 上下文注入
            dep_context = _build_dependency_context(task, state)
            if dep_context:
                t_input["_context"] = dep_context

            # Skill context injection
            skill_system_prompt = state.get("skill_system_prompt", "")
            if skill_system_prompt:
                t_input["_skill_system_prompt"] = skill_system_prompt
                t_input["_skill_name"] = state.get("skill_name", "")

            # 平台上下文注入（来自前台 PlatformSelector 选择或文本检测）
            context = state.get("context", {})
            platform_code = context.get("platform", "")
            if platform_code:
                t_input["platform"] = platform_code
                if context.get("platform_system_prompt"):
                    t_input["_platform_system_prompt"] = context["platform_system_prompt"]
                if context.get("platform_writing_rules"):
                    t_input["_platform_writing_rules"] = context["platform_writing_rules"]
                if context.get("platform_quality_check"):
                    t_input["_platform_quality_check"] = context["platform_quality_check"]
                if context.get("platform_agent_pipeline"):
                    t_input["_platform_agent_pipeline"] = context["platform_agent_pipeline"]
                # content_type 回退映射
                if "content_type" not in t_input and context.get("platform"):
                    from .platform_profile import get_profile
                    profile = get_profile(platform_code)
                    if profile:
                        t_input["content_type"] = platform_code

            # MCP 工具
            mcp_tools = state.get("_mcp_tools", [])
            if mcp_tools:
                t_input["_mcp_tools"] = mcp_tools

            # 缓存检查
            cacheable_types = {"SEOAgent", "ReviewAgent", "WebSearch"}
            use_cache = agent_type in cacheable_types and not t_input.get("_skip_cache")
            if use_cache:
                cached = await agent_cache.get(agent_type, t_input)
                if cached:
                    elapsed = round(time.time() - start_time, 2)
                    cached["time_elapsed"] = elapsed
                    cached["_from_cache"] = True
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

            # 任务开始：发布 thinking_step
            await _publish_streaming_event(state, "thinking_step", {
                "step_number": task_step_number,
                "agent": agent_type,
                "action": "开始执行",
                "reasoning": f"Agent {agent_type} 开始处理",
                "input_summary": str(t_input)[:200],
                "output_summary": "",
                "tokens_used": 0,
                "time_elapsed": 0,
                "status": "running",
            })

            # 流式 LLM 回调（安全版：catch 异常防止静默丢失）
            if streaming_mode and agent_type in {"WritingAgent", "SEOAgent", "ReviewAgent"}:
                async def _safe_publish(event_type: str, data: dict) -> None:
                    try:
                        await _publish_streaming_event(state, event_type, data)
                    except Exception as e:
                        logger.warning("流式事件发布失败", event_type=event_type, error=str(e))

                def _make_on_llm_chunk(agent: str):
                    async def _on_llm_chunk(chunk: str) -> None:
                        await _safe_publish("llm_chunk", {"agent": agent, "content": chunk, "timestamp": time.time()})
                    return _on_llm_chunk
                def _make_on_reasoning(agent: str):
                    async def _on_reasoning(text: str) -> None:
                        await _safe_publish("reasoning", {"agent": agent, "content": text, "timestamp": time.time()})
                    return _on_reasoning
                t_input["_on_llm_chunk"] = _make_on_llm_chunk(agent_type)
                t_input["_on_reasoning"] = _make_on_reasoning(agent_type)

            # 执行 handler
            _mcp = mcp_tools if mcp_tools else t_input.get("_mcp_tools")
            result = await handler(t_input, llm, _mcp)

            # 清理不可序列化的回调引用
            t_input.pop("_on_llm_chunk", None)
            t_input.pop("_on_reasoning", None)

            elapsed = round(time.time() - start_time, 2)
            result["time_elapsed"] = elapsed
            if "thinking_steps" not in result:
                result["thinking_steps"] = []

            # 任务完成事件
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

            # 缓存结果
            if use_cache and result.get("status") in ("completed", "partial"):
                try:
                    _cache_input = {k: v for k, v in t_input.items() if not k.startswith('_')}
                    await agent_cache.set(agent_type, _cache_input, result)
                except Exception as e:
                    logger.warning("缓存结果失败", agent_type=agent_type, error=str(e))

            return result

        except ToolExecutionError as e:
            elapsed = round(time.time() - start_time, 2)
            logger.error(
                "MCP 工具执行失败",
                agent_type=agent_type,
                tool_name=e.tool_name,
                error=str(e),
                _type="agent",
                event_name="tool_execution_error",
                failure_type="mcp_tool",
                error_code=e.error_code,
            )
            await _publish_streaming_event(state, "thinking_step", {
                "step_number": task_step_number,
                "agent": agent_type,
                "action": "工具调用失败",
                "reasoning": f"工具 {e.tool_name} 执行失败: {e.message}",
                "input_summary": str(task_input)[:200],
                "output_summary": str(e),
                "tokens_used": 0,
                "time_elapsed": elapsed,
                "status": "failed",
            })
            return {
                "status": "failed",
                "agent_type": agent_type,
                "failure_type": "mcp_tool",
                "error_code": e.error_code,
                "tool_name": e.tool_name,
                "error": str(e),
                "error_details": e.details,
                "time_elapsed": elapsed,
                "thinking_steps": [{"agent": agent_type, "action": "tool_error", "tool": e.tool_name, "reasoning": e.message, "status": "failed"}],
            }

        except HandlerError as e:
            elapsed = round(time.time() - start_time, 2)
            logger.error(
                "Handler 执行失败",
                agent_type=agent_type,
                handler=e.handler,
                error=str(e),
                _type="agent",
                event_name="handler_error",
                failure_type="handler",
                error_code=e.error_code,
            )
            await _publish_streaming_event(state, "thinking_step", {
                "step_number": task_step_number,
                "agent": agent_type,
                "action": "handler 失败",
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
                "failure_type": "handler",
                "error_code": e.error_code,
                "handler": e.handler,
                "error": str(e),
                "error_details": e.details,
                "time_elapsed": elapsed,
                "thinking_steps": [{"agent": agent_type, "action": "handler_error", "reasoning": str(e), "status": "failed"}],
            }

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            exc_type = type(e).__name__
            logger.error(
                "Agent 执行失败（未知异常）",
                agent_type=agent_type,
                error=str(e),
                exc_info=True,
                _type="agent",
                event_name="agent_error",
                failure_type="unknown",
                error_type=exc_type,
            )
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
                "failure_type": "unknown",
                "error_type": exc_type,
                "error": str(e),
                "time_elapsed": elapsed,
                "thinking_steps": [{"agent": agent_type, "action": "error", "reasoning": str(e), "status": "failed"}],
            }

    return await handle_fallback({"agent_type": agent_type, "input": task_input}, llm, None)


def _truncate_text(text: str, max_chars: int = 2000) -> str:
    """截断文本到 max_chars 字符，防止 Context Bleeding"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[...共{len(text)}字，已截断显示前{max_chars}字]"


def _build_dependency_context(task: dict, state: AgentGraphState) -> dict:
    """构建依赖上下文（只传递 content 字段，防止 Context Bleeding）

    每个依赖的内容截断到 2000 字符以内。
    """
    task_queue = state.get("task_queue", [])
    task_results = state.get("task_results", {})
    context = {}
    dep_count = len(task.get("dependencies", []))
    # 每个依赖的预算 = min(2000, 6000 / dep_count)
    budget_per_dep = max(1000, 6000 // max(dep_count, 1))

    for dep in task.get("dependencies", []):
        dep_id = _resolve_dep_id(dep, task_queue)
        if dep_id is None:
            continue
        if dep_id in task_results:
            prev = task_results[dep_id]
            output = prev.get("output", {})
            if isinstance(output, dict):
                content = output.get("content", "")
            elif isinstance(output, str):
                content = output
            elif isinstance(output, (list, tuple)):
                # 列表/元组取第一个元素的 content
                content = ""
                for item in output:
                    if isinstance(item, dict):
                        content = item.get("content", item.get("text", ""))
                    elif isinstance(item, str):
                        content = item
                    if content:
                        break
            else:
                # 非标准类型：只取前 200 字符的 repr，避免 str() 整个大对象
                content = repr(output)[:200] if output else ""
            context[dep_id] = {
                "content": _truncate_text(content, budget_per_dep),
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
        kind="INTERNAL",
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

        _CONTENT_TYPES = {
            "text", "blog", "social_post",
            "xhs_post", "wechat_article", "zhihu_article",
            "fiction", "storyboard", "script",
            "copywriting", "newsletter",
            "technical_doc", "tutorial",
        }

        primary_content = ""
        review_content = ""
        meta_sections = {}

        # 按优先级收集候选内容：SEO改写 > WritingAgent > 其他
        content_candidates = []
        for task in task_queue:
            tid = task.get("id", "")
            result = task_results.get(tid, {})
            output = result.get("output", {})
            agent_type = result.get("agent_type", task.get("agent_type", ""))

            if agent_type == "ReviewAgent" and isinstance(output, dict):
                # 审核内容单独收集，不进入主正文
                review_content = output.get("content", "")

            if agent_type == "SEOAgent" and isinstance(output, dict):
                metadata = output.get("metadata", {})
                if metadata:
                    meta_sections["seo"] = metadata
                seo_content = output.get("content", "")
                if seo_content and output.get("type") != "metadata":
                    content_candidates.append(("seo_rewrite", seo_content))

            if agent_type == "WritingAgent" and isinstance(output, dict):
                content = output.get("content", "")
                content_type = output.get("content_type", "")
                output_type = output.get("type", "")
                if content and (content_type in _CONTENT_TYPES or output_type == "text"):
                    content_candidates.append(("writing", content))
            elif agent_type == "WritingAgent" and isinstance(output, str) and output:
                content_candidates.append(("writing", output))

        # 按优先级选择主内容：seo_rewrite > writing
        for priority in ("seo_rewrite", "writing"):
            for ptype, content in content_candidates:
                if ptype == priority:
                    primary_content = content
                    break
            if primary_content:
                break

        if not primary_content:
            logger.warning("未找到主内容，退回到拼接方式，但跳过审核 Agent")
            for task in task_queue:
                tid = task.get("id", "")
                result = task_results.get(tid, {})
                agent_type = result.get("agent_type", task.get("agent_type", ""))
                # 跳过审核 agent，不进入主正文
                if agent_type == "ReviewAgent":
                    if not review_content:
                        output = result.get("output", {})
                        review_content = output.get("content", "") if isinstance(output, dict) else ""
                    continue
                output = result.get("output", {})
                if isinstance(output, dict):
                    content = output.get("content", "")
                else:
                    content = str(output) if output else ""
                if content:
                    full_text_parts.append(content)
            primary_content = "".join(full_text_parts)
        else:
            full_text_parts = [primary_content]

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

        completed = len(completed_ids)
        failed = len(failed_ids)

        status = "completed"
        if failed > 0 and completed == 0:
            status = "failed"
        elif failed > 0:
            status = "partial"

        asyncio.create_task(_persist_results_async(state, sections, full_text_parts, status))

        merge_span.set_attribute("merge.status", status)
        merge_span.set_status("OK")

        return {
            "final_output": {
                "goal": plan.get("goal", ""),
                "plan_reasoning": plan.get("reasoning", ""),
                "platform": plan.get("platform", ""),
                "sections": sections,
                "full_text": "".join(full_text_parts),
                "review_content": review_content,  # 审核内容独立字段，不混入 full_text
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

        save_agent_task(
            task_id=task_id or state.get("trace_task_id", ""),
            user_id=state.get("context", {}).get("user_id", ""),
            prompt=state.get("user_request", ""),
            status=status,
            result={"sections": sections, "full_text": "".join(full_text_parts)},
            total_steps=len(state.get("task_queue", [])),
            completed_steps=len(state.get("completed_task_ids", [])),
            total_tokens=sum(s.get("tokens_used", 0) for s in state.get("task_results", {}).values()),
            agent_events={"events": state.get("thinking_steps", [])},
        )

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
        logger.warning("持久化失败", error=str(e))


async def _persist_results_async(state: AgentGraphState, sections: list, full_text_parts: list, status: str) -> None:
    """异步持久化执行结果到数据库"""
    try:
        await asyncio.to_thread(_persist_results_sync, state, sections, full_text_parts, status)
    except Exception as e:
        logger.warning("异步持久化失败", error=str(e))


# ──────────────────────────────────────────────
# 路由函数（用于条件边）
# ──────────────────────────────────────────────

def should_continue_or_finish(state: AgentGraphState) -> str:
    """条件边路由函数：判断是否继续执行任务

    返回值：
    - "execute": 有可执行的任务，继续执行
    - "done": 所有任务已完成或死锁
    """
    task_queue = state.get("task_queue", [])
    completed = len(state.get("completed_task_ids", []))
    failed = len(state.get("failed_task_ids", []))
    ready = state.get("ready_tasks", [])
    total_tasks = len(task_queue)
    total_done = completed + failed

    if ready:
        return "execute"
    if total_done >= total_tasks:
        return "done"
    # 有未完成任务但没有就绪任务 → 死锁或依赖链断裂
    logger.warning("没有就绪任务但未全部完成，强制退出防止死锁",
                   total_tasks=total_tasks, total_done=total_done)
    return "done"
