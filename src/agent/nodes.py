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
import re
import time
import uuid
from typing import Any

import structlog
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .state import AgentGraphState
from .agent_utils import (
    UNDERSTANDING_SYSTEM_PROMPT, PLANNING_SYSTEM_PROMPT,
)
from .platform_registry import PLATFORMS, detect_platform, get_names_map
from .handlers import HANDLER_MAP, handle_fallback
from .cache import agent_cache
from .stage_map import get_stage_label, get_stage_icon
from .registry import get as get_card, by_category, CATEGORY_DEPENDENCY_RULES

from src.core.exceptions import HandlerError, ToolExecutionError
from src.observability.metrics import record_agent_task, record_agent_run
from src.observability.agent_tracer import get_agent_tracer, AgentTracer
from src.logging_shared.agent_events import AgentEvent, TokenUsage, EventType
from src.logging_shared.agent_emitter import get_agent_event_emitter

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


def _build_fallback_plan(user_request: str, platform_code: str | None = None) -> dict:
    """简化兜底：纯关键词匹配，无 intent_analysis 依赖"""
    goal = user_request[:200]
    text = user_request.lower()

    detected = platform_code or detect_platform(user_request)
    p = PLATFORMS.get(detected) if detected else None
    content_type = p.content_type if p else "blog"

    # 简易关键词 → Agent 映射
    if any(kw in text for kw in ["代码", "脚本", "程序", "写一个", "python", "java", "javascript", "function"]):
        suggested = ["code", "review"]
    elif any(kw in text for kw in ["图表", "架构图", "流程图", "mermaid", "diagram"]):
        suggested = ["writing", "diagram", "review"]
    elif any(kw in text for kw in ["搜索", "热点", "热搜", "查", "最新", "趋势", "news", "trending"]):
        suggested = ["search", "analysis", "writing", "review", "fix"]
    elif any(kw in text for kw in ["分析", "url", "链接", "网站", "网页", "抓取"]):
        suggested = ["search", "analysis", "writing", "review"]
    else:
        suggested = ["writing", "review", "fix"]

    tasks = []
    last_id_index = -1

    # 兜底计划：按类别从 registry 获取 Agent 类型名（避免硬编码）
    # short_name = category name, input_template 仍是上下文相关的
    _CATEGORY_INPUTS: dict[str, tuple[str, dict]] = {
        "writing": ("high", {"topic": user_request, "content_type": content_type, "platform": detected or ""}),
        "review": ("medium", {"topic": user_request, "content_type": content_type, "platform": detected or ""}),
        "fix":    ("medium", {"topic": user_request, "content_type": content_type, "platform": detected or ""}),
        "search": ("low",    {"query": user_request, "count": 5}),
        "analysis":("medium", {"topic": user_request, "platform": detected or ""}),
        "code":   ("medium", {"task": user_request}),
        "diagram":("low",    {"topic": user_request}),
    }

    built: list[dict] = []
    for short_name in suggested:
        if short_name not in _CATEGORY_INPUTS:
            continue
        cards = by_category(short_name)  # type: ignore[arg-type]
        if not cards:
            continue
        agent_type = cards[0].name
        priority, input_data = _CATEGORY_INPUTS[short_name]

        task = {
            "id": str(uuid.uuid4()), "agent_type": agent_type,
            "input": {**input_data, "platform": detected or ""},
            "dependencies": [],
            "can_parallel_with": [],
            "retry_on_low_quality": agent_type in ("WritingAgent", "FixAgent"),
            "priority": priority,
        }
        built.append(task)

    # 自动构建依赖链：从 registry 的 CATEGORY_DEPENDENCY_RULES 推断
    _cat_task_ids: dict[str, list[str]] = {}
    for t in built:
        card = get_card(t["agent_type"])
        cat = card.category if card else ""
        _cat_task_ids.setdefault(cat, []).append(t["id"])

    for t in built:
        card = get_card(t["agent_type"])
        if not card:
            continue
        deps = t.setdefault("dependencies", [])
        dep_cats = CATEGORY_DEPENDENCY_RULES.get(card.category, [])
        for dep_cat in dep_cats:
            dep_ids = _cat_task_ids.get(dep_cat, [])
            if dep_ids and dep_ids[-1] not in deps:
                deps.append(dep_ids[-1])

    agent_names = [t["agent_type"] for t in built]
    platform_hint = f"（{p.name}）" if p else ""
    return {
        "goal": goal,
        "tasks": built,
        "reasoning": f"平台计划{platform_hint}：{' → '.join(agent_names)}",
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
            elif event_type == "stage":
                await task_queue.publish_stage(task_id, data)
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


# ── 最小规则兜底：仅在 LLM 理解完全失败时使用 ──

def _build_fallback_understanding(user_request: str, context: dict) -> dict:
    """最小规则兜底：LLM 理解失败时，纯关键词检测语言和平台"""
    text = user_request.strip()
    has_chinese = bool(re.search(r'[一-鿿]', text))

    from .platform_registry import detect_platform, get_names_map
    platform = context.get("platform") or detect_platform(text)
    platform_name = get_names_map().get(platform, "") if platform else ""

    return {
        "intent": {"coarse": "CREATE", "fine": "generate content"},
        "slots": {
            "content_type": "文章",
            "domain": "其他",
            "creativity_level": "medium",
            "length_hint": "medium",
            "tone": None,
            "audience": None,
        },
        "missing_slots": [],
        "cleaned_prompt": user_request,
        "safety": True,
        "safety_reason": "",
        "platform": platform,
        "platform_name": platform_name,
        "language": "zh" if has_chinese else "en",
    }


# ──────────────────────────────────────────────
# 1. understand_intent 节点
# ──────────────────────────────────────────────

async def understand_intent(state: AgentGraphState) -> dict:
    """理解 Agent：纯 LLM 意图分析，无规则偏差"""
    tracer = _get_tracer()
    user_request = state.get("user_request", "")
    context = state.get("context", {})
    trace_id = str(uuid.uuid4())

    # 从上下文获取平台标签（前端可选提供）
    platform_tag = context.get("platform", "")
    platform_context = ""
    if platform_tag:
        from .platform_registry import PLATFORMS as _PLATFORMS
        meta = _PLATFORMS.get(platform_tag)
        if meta:
            platform_context = f"\nPlatform tag from UI: {meta.name} ({platform_tag})"

    with tracer.start_as_current_span(
        "agent.understand",
        kind="INTERNAL",
        attributes={
            "agent.node": "understand_intent",
            "agent.user_request.preview": user_request[:200],
            "custom.run_id": trace_id,
        },
    ) as span:
        llm = _get_llm(state, "ManagerAgent")
        understanding = None
        understanding_raw = ""

        if llm:
            try:
                response = await asyncio.wait_for(
                    llm.ainvoke([
                        SystemMessage(content=UNDERSTANDING_SYSTEM_PROMPT),
                        HumanMessage(content=f"User request: {user_request}{platform_context}"),
                    ]),
                    timeout=30.0,
                )
                understanding_raw = response.content
                parsed = json.loads(response.content)
                # Validate required fields
                if isinstance(parsed, dict) and all(k in parsed for k in ("intent", "cleaned_prompt", "safety")):
                    understanding = parsed
                else:
                    raise ValueError("Missing required fields in understanding response")
            except Exception as e:
                logger.warning("LLM 理解失败，使用规则兜底", error=str(e)[:200])
                span.record_exception(e)

        # LLM 完全失败时：最小规则兜底
        if not understanding:
            understanding = _build_fallback_understanding(user_request, context)

        updates = {
            "trace_task_id": trace_id,
            "understanding": understanding,
            "understanding_raw": understanding_raw,
            "start_time": time.time(),
            "thinking_steps": [],
            "status": "running",
        }

        updates.update(_add_thinking_step(
            state, "UnderstandingAgent", "意图分析完成",
            reasoning=f"意图={understanding['intent']['coarse']} - {understanding['intent']['fine']}，平台={understanding.get('platform_name', '无')}，安全={understanding['safety']}",
            input_summary=user_request[:200],
            output_summary=f"清理后：{understanding['cleaned_prompt'][:100]}",
        ))

        span.set_attribute("agent.intent", understanding.get("intent", {}).get("coarse", "unknown"))
        span.set_attribute("agent.safety", str(understanding["safety"]))
        span.set_status("OK")

        # Emit stage 事件
        await _publish_streaming_event(state, "stage", {
            "agent": "ManagerAgent",
            "label": "思考分析中",
            "icon": "🧠",
            "status": "started",
            "timestamp": time.time(),
        })

        # Emit reasoning 事件（简要总结，不发原始 JSON）
        intent_info = understanding.get("intent", {})
        slots = understanding.get("slots", {})
        platform_name = understanding.get("platform_name", "")
        summary = f"{intent_info.get('coarse', '')}：{intent_info.get('fine', '')}"
        if platform_name:
            summary += f"（平台：{platform_name}）"
        if slots.get("tone"):
            summary += f"，风格：{slots['tone']}"
        if slots.get("length_hint"):
            summary += f"，篇幅：{slots['length_hint']}"
        await _publish_streaming_event(state, "reasoning", {
            "agent": "ManagerAgent",
            "content": summary,
            "timestamp": time.time(),
        })

        await _publish_streaming_event(state, "stage", {
            "agent": "ManagerAgent",
            "status": "completed",
            "duration": time.time() - (state.get("start_time", time.time()) or time.time()),
            "timestamp": time.time(),
        })

        return updates


# ──────────────────────────────────────────────
# 2. plan_tasks 节点
# ──────────────────────────────────────────────

def _parse_plan_from_response(response: str, goal: str, platform: str | None = None) -> dict | None:
    """从 LLM 响应中提取 JSON 计划（简化版，无严格 Agent 类型验证）"""
    import json as _json, re as _re

    content = response.strip()
    code_block_match = _re.search(r"```(?:json)?\s*([\s\S]+?)```", content)
    if code_block_match:
        content = code_block_match.group(1)

    try:
        plan_dict = _json.loads(content)
    except _json.JSONDecodeError:
        json_match = _re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                plan_dict = _json.loads(json_match.group(0))
            except _json.JSONDecodeError:
                return None
        else:
            return None

    # 支持两种格式：{"tasks": [...]} 或直接数组 [{...}, {...}]
    raw_tasks = []
    if isinstance(plan_dict, list):
        raw_tasks = plan_dict
    elif isinstance(plan_dict, dict):
        raw_tasks = plan_dict.get("tasks", [])
    if not raw_tasks:
        return None

    tasks = []
    for i, t in enumerate(raw_tasks):
        if not isinstance(t, dict):
            continue
        agent_type = t.get("agent_type") or t.get("agent")
        if not agent_type:
            continue
        task_input = t.get("input") or {}
        if not task_input and t.get("task"):
            task_input = {"topic": t["task"], "content_type": "blog"}
        tasks.append({
            "id": str(uuid.uuid4()),
            "agent_type": agent_type,
            "input": task_input,
            "dependencies": t.get("dependencies", []),
            "can_parallel_with": t.get("can_parallel_with", []),
            "retry_on_low_quality": t.get("retry_on_low_quality", False),
            "priority": t.get("priority", "medium"),
            "config": {"reasoning": t.get("reasoning", ""), "priority": i},
        })

    if not tasks:
        return None

    return {
        "goal": plan_dict.get("goal", goal) if isinstance(plan_dict, dict) else goal,
        "tasks": tasks,
        "reasoning": plan_dict.get("reasoning", "") if isinstance(plan_dict, dict) else "",
        "estimated_steps": len(tasks),
        "plan_method": "llm",
    }


async def plan_tasks(state: AgentGraphState) -> dict:
    """规划 Agent：仅根据 understanding 自治分解为子任务，无 suggested_agents 偏见"""
    tracer = _get_tracer()
    understanding = state.get("understanding", {})
    user_request = state.get("user_request", "")
    trace_id = state.get("trace_task_id", str(uuid.uuid4()))
    cleaned_prompt = understanding.get("cleaned_prompt", user_request)
    detected_platform = understanding.get("platform")

    # 预计算计划：跳过 LLM（由 Orchestrator 传入）
    precomputed = state.get("plan")
    if precomputed and precomputed.get("tasks"):
        logger.info("使用预计算计划，跳过 LLM 规划")
        return {
            "plan": precomputed,
            "plan_reasoning": precomputed.get("reasoning", ""),
            "task_queue": precomputed.get("tasks", []),
        }

    platform_hint = f"（{understanding.get('platform_name', '')}）" if detected_platform else ""

    with tracer.start_as_current_span(
        "agent.plan",
        kind="INTERNAL",
        attributes={
            "agent.node": "plan_tasks",
            "agent.intent": understanding.get("intent", "unknown"),
            "agent.platform": detected_platform or "none",
            "custom.run_id": trace_id,
        },
    ) as span:
        context_str = json.dumps({
            "intent": understanding.get("intent", ""),
            "cleaned_prompt": cleaned_prompt[:500],
            "content_type": understanding.get("content_type", "blog"),
            "platform": detected_platform,
            "platform_name": understanding.get("platform_name", ""),
            "language": understanding.get("language", "zh"),
        }, ensure_ascii=False, indent=2)

        plan = None
        llm = _get_llm(state, "ManagerAgent")

        if llm:
            try:
                response = await asyncio.wait_for(
                    llm.ainvoke([
                        SystemMessage(content=PLANNING_SYSTEM_PROMPT),
                        HumanMessage(content=f"""## User Request
{cleaned_prompt}

## Intent Understanding
{context_str}

Based on the above understanding, select the most appropriate agents to complete the task."""),
                    ]),
                    timeout=60.0,
                )
                plan = _parse_plan_from_response(response.content, cleaned_prompt, platform=detected_platform)

                if not plan or len(plan.get("tasks", [])) == 0:
                    logger.warning("LLM 计划没有任务，使用兜底方案. LLM响应: %s", response.content[:300] if response.content else "空")
                    plan = _build_fallback_plan(cleaned_prompt, detected_platform)
                else:
                    plan["plan_method"] = "llm"
                    logger.info("LLM 计划已生成：%d 个任务", len(plan.get("tasks", [])))

            except Exception as e:
                logger.warning("LLM 规划失败，使用兜底方案", error=str(e)[:200])
                span.record_exception(e)

        if not plan:
            plan = _build_fallback_plan(cleaned_prompt, detected_platform)


        # 注入理解 Agent 的结构化槽位到每个任务
        slots = understanding.get("slots", {})
        if slots:
            for task in plan.get("tasks", []):
                task_input = task.get("input", {})
                for key in ("tone", "audience", "creativity_level", "length_hint", "domain", "content_type"):
                    val = slots.get(key)
                    if val and val not in ("null", None) and key not in task_input:
                        task_input[key] = val
                if understanding.get("language"):
                    task_input.setdefault("language", understanding["language"])

        # 自动修正依赖链（防止 LLM 漏设依赖导致并行错误）
        # 从 registry 的 CATEGORY_DEPENDENCY_RULES 推断
        tasks = plan.get("tasks", [])

        # 按类别分组 task_id
        _category_task_ids: dict[str, list[str]] = {}
        for t in tasks:
            card = get_card(t.get("agent_type", ""))
            cat = card.category if card else ""
            _category_task_ids.setdefault(cat, []).append(t["id"])

        for t in tasks:
            card = get_card(t.get("agent_type", ""))
            if not card:
                continue
            deps = t.setdefault("dependencies", [])
            dep_cats = CATEGORY_DEPENDENCY_RULES.get(card.category, [])
            for dep_cat in dep_cats:
                dep_ids = _category_task_ids.get(dep_cat, [])
                if dep_ids and dep_ids[-1] not in deps:
                    deps.append(dep_ids[-1])

        updates = {
            "plan": plan,
            "plan_reasoning": plan.get("reasoning", ""),
            "task_queue": plan.get("tasks", []),
        }

        updates.update(_add_thinking_step(
            state, "PlanningAgent", "任务拆解完成",
            reasoning=f"{plan.get('plan_method', 'fallback')} 规划完成，共 {len(plan.get('tasks', []))} 个任务{platform_hint}",
            input_summary=cleaned_prompt[:200],
            output_summary=f"执行计划：{plan.get('reasoning', '')[:80]}",
        ))

        span.add_event("plan_built", {
            "task_count": len(plan.get("tasks", [])),
            "method": plan.get("plan_method", "fallback"),
        })
        span.set_attribute("plan.method", plan.get("plan_method", "fallback"))
        span.set_attribute("agent.task_count", len(plan.get("tasks", [])))
        span.set_status("OK")

        # Emit stage 事件 for PlanningAgent
        await _publish_streaming_event(state, "stage", {
            "agent": "PlanningAgent",
            "label": "任务规划中",
            "icon": "📋",
            "status": "started",
            "timestamp": time.time(),
        })

        # 不 emit 硬编码 reasoning，thinking_step 已包含规划信息

        await _publish_streaming_event(state, "stage", {
            "agent": "PlanningAgent",
            "status": "completed",
            "duration": time.time() - (state.get("start_time", time.time()) or time.time()),
            "timestamp": time.time(),
        })

        return updates


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
                    "_loop_created_in_cycle": False,
                }
            else:
                span.add_event("waiting_for_dependencies", {"deps_unmet": deps_unmet})
                span.set_status("OK")
                return {"ready_tasks": [], "_loop_created_in_cycle": False}

        span.set_status("OK")
        return {"ready_tasks": ready, "_loop_created_in_cycle": False}


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
        # ── 如果上一轮审核已通过，跳过本轮待执行的 FixAgent 循环任务 ──
        # 记录跳过的任务，等 task_results / completed 初始化后补充标记
        _skipped_tasks: list[dict] = []
        _is_review_passed = state.get("_last_review_passed", False)
        if _is_review_passed:
            filtered = []
            for t in ready_tasks:
                if t.get("agent_type") == "FixAgent" and t.get("input", {}).get("_review_round", 0) > 0:
                    _skipped_tasks.append(t)
                    logger.info("审核已通过，跳过 FixAgent（第 %d 轮）", t["input"].get("_review_round", 0))
                else:
                    filtered.append(t)
            if _skipped_tasks:
                ready_tasks = filtered
                logger.info("本轮过滤掉 %d 个 FixAgent 任务", len(_skipped_tasks))

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

        # ── 将本轮跳过的 FixAgent 标记为已完成（避免 schedule 死循环） ──
        for st in _skipped_tasks:
            sid = st.get("id", "")
            if sid not in task_results:
                task_results[sid] = {
                    "status": "completed", "agent_type": "FixAgent",
                    "output": {"content": "", "passed": True, "skipped": True},
                    "thinking_steps": [{
                        "agent": "FixAgent", "action": "跳过",
                        "reasoning": "审核已通过，无需再修改",
                        "output_summary": "跳过",
                        "status": "done",
                    }],
                }
            if sid not in completed:
                completed.append(sid)

        # 检查本轮完成的 ReviewAgent 是否通过了审核
        # 用于短路后续 review-fix 循环，避免质量已达标时继续浪费 Token
        # 注意：只有 re-review 轮次（_review_round > 0）的通过才算数，
        # 初审（round 0）的格式可靠性不如复审（LLM 可能遗漏结论行），
        # 如果让它阻塞循环，FixAgent 完成后就不会追加复审+再润色。
        _last_review_passed = state.get("_last_review_passed", False)
        task_queue = state.get("task_queue", [])
        for task_id, result in task_results.items():
            if result.get("agent_type") != "ReviewAgent":
                continue
            if result.get("status") not in ("completed", "partial"):
                continue
            task_def = next((t for t in task_queue if t.get("id") == task_id), None)
            review_round = task_def.get("input", {}).get("_review_round", 0) if task_def else 0
            if review_round == 0:
                continue  # 初审不阻塞循环，由 re-review 决定是否停止
            output = result.get("output", {})
            if isinstance(output, dict) and output.get("passed") is True:
                _last_review_passed = True

        # ── Review-Fix 循环检测（最多 2 轮，每批只允许生成一轮，防止指数爆炸） ──
        _MAX_REVIEW_LOOPS = 2
        _review_fix_cycles = state.get("_review_fix_cycles", 0)
        _loop_already_created = state.get("_loop_created_in_cycle", False)

        loop_new_tasks: list[dict] = []
        if _review_fix_cycles < _MAX_REVIEW_LOOPS and not _loop_already_created and not _last_review_passed:
            for tid in list(completed):
                result = task_results.get(tid, {})
                if result.get("agent_type") != "FixAgent":
                    continue

                # 每次润色后自动追加复审+再润色（不依赖旧审核结果）
                # 防止同一批多个 FixAgent → 生成多个配对 → 指数爆炸
                fix_task = next((t for t in state.get("task_queue", []) if t.get("id") == tid), None)
                if not fix_task:
                    continue

                loop_round = _review_fix_cycles + 1
                new_review_id = str(uuid.uuid4())
                new_fix_id = str(uuid.uuid4())
                topic = fix_task.get("input", {}).get("topic", "")

                loop_new_tasks.append({
                    "id": new_review_id,
                    "agent_type": "ReviewAgent",
                    "input": {"topic": topic, "_skip_cache": True, "_review_round": loop_round},
                    "dependencies": [tid],
                    "config": {"priority": 0, "reasoning": f"复审(第{loop_round}轮)"},
                })
                loop_new_tasks.append({
                    "id": new_fix_id,
                    "agent_type": "FixAgent",
                    "input": {"topic": topic, "_skip_cache": True, "_review_round": loop_round},
                    "dependencies": [tid, new_review_id],
                    "config": {"priority": 0, "reasoning": f"再润色(第{loop_round}轮)"},
                })
                logger.info("润色完成，自动追加第 %d 轮复审+再润色", loop_round)
                break  # 每批只生成一轮，防止指数爆炸

        existing = state.get("thinking_steps", [])
        batch_span.add_event("batch_completed", {
            "completed": len(completed), "failed": len(failed),
            "review_loop_created": len(loop_new_tasks) > 0,
        })
        batch_span.set_status("OK")

        record_agent_task(
            agent_type="batch",
            task_type="batch_execute",
            status="completed",
            duration_s=(time.time() - state.get("start_time", time.time())),
        )

        updates = {
            "task_results": task_results,
            "completed_task_ids": completed,
            "failed_task_ids": failed,
            "thinking_steps": existing + new_steps,
            "_last_review_passed": _last_review_passed,
        }
        if loop_new_tasks:
            updated_queue = list(state.get("task_queue", [])) + loop_new_tasks
            updates["task_queue"] = updated_queue
            updates["_review_fix_cycles"] = _review_fix_cycles + 1
            updates["_loop_created_in_cycle"] = True

        return updates


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

            # 理解槽位注入（tone、audience 等结构化参数）
            understanding = state.get("understanding", {})
            slots = understanding.get("slots", {}) if understanding else {}
            for key in ("tone", "audience", "creativity_level", "length_hint", "domain"):
                val = slots.get(key) if slots else None
                if val and val not in ("null", None) and key not in t_input:
                    t_input[key] = val
            if understanding and understanding.get("language"):
                t_input.setdefault("language", understanding["language"])

            # MCP 工具
            mcp_tools = state.get("_mcp_tools", [])
            if mcp_tools:
                t_input["_mcp_tools"] = mcp_tools

            # Emit stage:started 事件（在缓存检查之前，确保前端 timeline 一致性）
            review_round = task_input.get("_review_round", 0)
            stage_data = {
                "agent": agent_type,
                "label": get_stage_label(agent_type),
                "icon": get_stage_icon(agent_type),
                "status": "started",
                "timestamp": time.time(),
            }
            if review_round > 0:
                stage_data["_review_round"] = review_round
            await _publish_streaming_event(state, "stage", stage_data)

            # 缓存检查
            cacheable_types = {"ReviewAgent", "WebSearch"}
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
                        "action": "命中缓存",
                        "reasoning": f"命中缓存，agent={agent_type}",
                        "input_summary": str(t_input)[:100],
                        "output_summary": cached.get("output", {}).get("content", "")[:200],
                        "tokens_used": cached.get("tokens_used", 0),
                        "time_elapsed": elapsed,
                        "status": "done",
                        "_from_cache": True,
                    })
                    # Emit stage:completed for cache hit too
                    await _publish_streaming_event(state, "stage", {
                        "agent": agent_type,
                        "status": "completed",
                        "duration": elapsed,
                        "timestamp": time.time(),
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
            if streaming_mode:
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

            # 执行 handler（带安全兜底）
            _mcp = mcp_tools if mcp_tools else t_input.get("_mcp_tools")

            # Emit agent_start 到控制台
            emitter = get_agent_event_emitter()
            await emitter.emit(AgentEvent(
                event_type="agent_start",
                agent_name=agent_type,
                message=str(task_input.get("topic", task_input.get("query", task_input.get("task", ""))))[:100] or "开始执行",
                run_id=task_id or agent_type,
                llm_model=getattr(llm, 'model_name', '') if llm else '',
            ))

            result = await _safe_run_handler(agent_type, handler, t_input, llm, _mcp, task=task, state=state)

            # 归一化为标准信封格式（兼容旧 handler）
            result = _normalize_agent_result(result, agent_type)

            # 清理不可序列化的回调引用
            t_input.pop("_on_llm_chunk", None)
            t_input.pop("_on_reasoning", None)

            elapsed = round(time.time() - start_time, 2)
            result["time_elapsed"] = elapsed

            # Emit stage:completed 事件
            await _publish_streaming_event(state, "stage", {
                "agent": agent_type,
                "status": "completed",
                "duration": elapsed,
                "timestamp": time.time(),
            })

            # 任务完成事件
            output_content = result.get("output", {}).get("content", "")
            await _publish_streaming_event(state, "thinking_step", {
                "step_number": task_step_number,
                "agent": agent_type,
                "action": "完成",
                "reasoning": f"Agent {agent_type} 执行完成",
                "input_summary": str(t_input)[:200],
                "output_summary": output_content[:300] if output_content else (result.get("error", "") or ""),
                "tokens_used": result.get("tokens_used", 0),
                "time_elapsed": elapsed,
                "status": result.get("status", "completed"),
            })

            # Emit agent_finish 到控制台
            await emitter.emit(AgentEvent(
                event_type="agent_finish",
                agent_name=agent_type,
                message=output_content[:100] or "完成",
                run_id=task_id or agent_type,
                duration_ms=elapsed * 1000,
                token_usage=TokenUsage(
                    prompt_tokens=result.get("tokens_used", 0),
                    completion_tokens=0,
                    total_tokens=result.get("tokens_used", 0),
                ),
                llm_model=getattr(llm, 'model_name', '') if llm else '',
            ))

            # 缓存结果
            if use_cache and result.get("status") in ("completed", "partial"):
                try:
                    _cache_input = {k: v for k, v in t_input.items() if not k.startswith('_')}
                    await agent_cache.set(agent_type, _cache_input, result)
                except Exception as e:
                    logger.warning("缓存结果失败", agent_type=agent_type, error=str(e))

            return result

        # handler 内的异常已由 _safe_run_handler 兜底，这里的 except
        # 只兜底归一化/日志/缓存等编排代码的意外错误
        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            logger.error(
                "Agent 编排异常",
                agent_type=agent_type,
                error=str(e),
                exc_info=True,
                _type="agent",
                event_name="orchestration_error",
            )
            await _publish_streaming_event(state, "thinking_step", {
                "step_number": task_step_number,
                "agent": agent_type,
                "action": "编排异常",
                "reasoning": str(e),
                "input_summary": str(task_input)[:200],
                "output_summary": str(e),
                "tokens_used": 0,
                "time_elapsed": elapsed,
                "status": "failed",
            })
            from .result import error_response
            return error_response(
                agent_type,
                error=f"编排异常: {e}",
                failure_type="orchestration",
                error_type=type(e).__name__,
                time_elapsed=elapsed,
            )

    return await handle_fallback({"agent_type": agent_type, "input": task_input}, llm, None)


def _find_review_dep(task: dict, task_queue: list[dict], task_results: dict) -> str | None:
    """在 FixAgent 的依赖中找 ReviewAgent 的 task_id"""
    for dep in task.get("dependencies", []):
        dep_id = _resolve_dep_id(dep, task_queue)
        if dep_id and dep_id in task_results:
            if task_results[dep_id].get("agent_type") == "ReviewAgent":
                return dep_id
    return None


def _normalize_agent_result(result: dict, agent_type: str) -> dict:
    """将 handler 返回结果归一化为标准信封格式

    允许 handler 逐步迁移到 agent_response()，迁移完成前兼容旧格式。
    标准格式: {"status","agent_type","output":{"content":str,"type":str}, "error","tokens_used","thinking_steps"}
    """
    output = result.get("output")

    # 情况 1: output 是字符串 → 包装为标准格式
    if isinstance(output, str):
        result["output"] = {"content": output, "type": "text"}

    # 情况 2: output 是列表 → 提取首个 content
    elif isinstance(output, (list, tuple)):
        content = ""
        for item in output:
            if isinstance(item, dict):
                content = item.get("content", item.get("text", ""))
            elif isinstance(item, str):
                content = item
            if content:
                break
        result["output"] = {"content": content, "type": "text", "_raw": output}

    # 情况 3: output 是 dict 但没有 content → 添加空 content
    elif isinstance(output, dict) and "content" not in output:
        result["output"] = {"content": "", "type": "metadata", **output}

    # 情况 4: 没有 output → 创建默认
    elif output is None:
        result["output"] = {"content": "", "type": "text"}

    # 确保必填字段存在
    result.setdefault("error", None)
    result.setdefault("tokens_used", 0)
    result.setdefault("prompt_tokens", 0)
    result.setdefault("completion_tokens", 0)
    if "thinking_steps" not in result:
        result["thinking_steps"] = []

    return result


async def _safe_run_handler(
    agent_type: str,
    handler,
    task_input: dict,
    llm,
    mcp_tools,
    *,
    task: dict,
    state: AgentGraphState,
) -> dict:
    """安全执行 handler，捕获所有异常返回标准错误响应。

    替代 _run_single_agent_task 中三个重复的 except 块。
    """
    from .result import error_response

    try:
        result = await handler(task_input, llm, mcp_tools)
        return result

    except ToolExecutionError as e:
        logger.error(
            "MCP 工具执行失败",
            agent_type=agent_type, tool_name=e.tool_name,
            error=str(e), _type="agent", event_name="tool_execution_error",
        )
        await _publish_streaming_event(state, "thinking_step", {
            "step_number": task.get("step", 0),
            "agent": agent_type,
            "action": "工具调用失败",
            "reasoning": f"工具 {e.tool_name} 执行失败: {e.message}",
            "input_summary": str(task_input)[:200],
            "output_summary": str(e),
            "tokens_used": 0, "time_elapsed": 0, "status": "failed",
        })
        return error_response(
            agent_type, error=str(e), failure_type="mcp_tool",
            tool_name=e.tool_name, error_code=e.error_code,
            error_details=e.details,
        )

    except HandlerError as e:
        logger.error(
            "Handler 执行失败",
            agent_type=agent_type, handler=e.handler,
            error=str(e), _type="agent", event_name="handler_error",
        )
        await _publish_streaming_event(state, "thinking_step", {
            "step_number": task.get("step", 0),
            "agent": agent_type, "action": "handler 失败",
            "reasoning": str(e),
            "input_summary": str(task_input)[:200],
            "output_summary": str(e),
            "tokens_used": 0, "time_elapsed": 0, "status": "failed",
        })
        return error_response(
            agent_type, error=str(e), failure_type="handler",
            error_code=e.error_code, handler=e.handler,
            error_details=e.details,
        )

    except Exception as e:
        logger.error(
            "Agent handler 异常",
            agent_type=agent_type, error=str(e), exc_info=True,
            _type="agent", event_name="agent_handler_error",
        )
        await _publish_streaming_event(state, "thinking_step", {
            "step_number": task.get("step", 0),
            "agent": agent_type, "action": "异常",
            "reasoning": str(e),
            "input_summary": str(task_input)[:200],
            "output_summary": str(e),
            "tokens_used": 0, "time_elapsed": 0, "status": "failed",
        })
        return error_response(
            agent_type, error=str(e), failure_type="unknown",
            error_type=type(e).__name__,
        )


def _truncate_text(text: str, max_chars: int = 2000) -> str:
    """截断文本到 max_chars 字符，防止 Context Bleeding"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[...共{len(text)}字，已截断显示前{max_chars}字]"


def _build_dependency_context(task: dict, state: AgentGraphState) -> dict:
    """构建依赖上下文（标准版）

    所有 handler 已统一使用 agent_response() 返回标准信封，
    output.content 永远存在，不再需要 isinstance 判断。
    每个依赖的内容截断到 2000 字符以内。
    """
    task_queue = state.get("task_queue", [])
    task_results = state.get("task_results", {})
    context = {}
    dep_count = len(task.get("dependencies", []))
    budget_per_dep = max(1000, 6000 // max(dep_count, 1))

    for dep in task.get("dependencies", []):
        dep_id = _resolve_dep_id(dep, task_queue)
        if dep_id is None:
            continue
        if dep_id in task_results:
            prev = task_results[dep_id]
            output = prev.get("output", {})
            content = (output.get("content", "") if isinstance(output, dict)
                       else str(output) if output else "")
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

        # 按优先级收集候选内容：FixAgent > ReviewFix > SEO改写 > WritingAgent
        content_candidates = []
        for task in task_queue:
            tid = task.get("id", "")
            result = task_results.get(tid, {})
            output = result.get("output", {})
            agent_type = result.get("agent_type", task.get("agent_type", ""))

            # FixAgent 的输出作为最高优先级最终内容
            if agent_type == "FixAgent" and isinstance(output, dict):
                content = output.get("content", "")
                if content:
                    content_candidates.append(("fix", content))

            if agent_type == "ReviewAgent" and isinstance(output, dict):
                review_content = output.get("content", "")
                fixed = output.get("fixed_content", "")
                if output.get("rewrite_applied") and fixed:
                    content_candidates.append(("review_fix", fixed))


            if agent_type == "WritingAgent" and isinstance(output, dict):
                content = output.get("content", "")
                content_type = output.get("content_type", "")
                output_type = output.get("type", "")
                if content and (content_type in _CONTENT_TYPES or output_type == "text"):
                    content_candidates.append(("writing", content))
            elif agent_type == "WritingAgent" and isinstance(output, str) and output:
                content_candidates.append(("writing", output))

        # 按优先级选择主内容：fix > review_fix > writing
        # review-fix 循环时取最后一轮（倒序查找）
        for priority in ("fix", "review_fix", "writing"):
            for ptype, content in reversed(content_candidates):
                if ptype == priority:
                    primary_content = content
                    break
            if primary_content:
                break

        if not primary_content:
            logger.warning("未找到主内容，退回到拼接方式，但跳过审查和中间 Agent")
            _SKIP_AGENTS_FALLBACK = {"ReviewAgent", "WebSearch", "URLCrawler", "HotTopicAgent"}
            for task in task_queue:
                tid = task.get("id", "")
                result = task_results.get(tid, {})
                agent_type = result.get("agent_type", task.get("agent_type", ""))
                if agent_type in _SKIP_AGENTS_FALLBACK:
                    if agent_type == "ReviewAgent":
                        if not review_content:
                            output = result.get("output", {})
                            review_content = output.get("content", "") if isinstance(output, dict) else ""
                        if isinstance(result.get("output"), dict):
                            fixed = result["output"].get("fixed_content", "")
                            if result["output"].get("rewrite_applied") and fixed:
                                full_text_parts.append(fixed)
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

        # 输出的 Agent 类型：只有内容产出型 Agent 进入预览
        _OUTPUT_AGENTS = {"WritingAgent", "ReviewAgent", "FixAgent", "CodeAgent", "DiagramAgent"}
        for task in task_queue:
            tid = task.get("id", "")
            result = task_results.get(tid, {})
            agent = result.get("agent_type", task.get("agent_type", ""))
            if agent not in _OUTPUT_AGENTS:
                continue
            if tid in completed_ids or tid in failed_ids:
                sections.append({
                    "agent": agent,
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

        # 清理：去掉 LLM 自行添加的"改进版"、"备选方案"等非正文段落
        raw_full_text = "".join(full_text_parts)
        clean_text = re.sub(
            r'【[^】]*改进[^】]*】[\s\S]*?(?=\n\n|\Z)',
            '', raw_full_text
        ).strip()

        return {
            "final_output": {
                "goal": plan.get("goal", ""),
                "plan_reasoning": plan.get("reasoning", ""),
                "platform": plan.get("platform", ""),
                "sections": sections,
                "full_text": clean_text,
                "review_content": review_content,
            },
            "status": status,
        }


def _persist_results_sync(state: AgentGraphState, sections: list, full_text_parts: list, status: str) -> None:
    """同步持久化执行结果到数据库"""
    try:
        from .persistence import save_agent_task, save_subtask, estimate_cost

        task_id = state.get("_async_task_id", "")
        if not task_id:
            return

        total_tokens = sum(s.get("tokens_used", 0) for s in state.get("task_results", {}).values())
        # 从 first task result 中取模型名用于成本估算
        first_result = next(iter(state.get("task_results", {}).values()), {})
        model_name = first_result.get("model", "") or first_result.get("llm_model", "")

        save_agent_task(
            task_id=task_id or state.get("trace_task_id", ""),
            user_id=state.get("context", {}).get("user_id", ""),
            prompt=state.get("user_request", ""),
            status=status,
            result={"sections": sections, "full_text": "".join(full_text_parts)},
            total_steps=len(state.get("task_queue", [])),
            completed_steps=len(state.get("completed_task_ids", [])),
            total_tokens=total_tokens,
            total_cost=estimate_cost(total_tokens, model_name),
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
