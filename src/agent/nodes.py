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
    _PLANNING_AGENTS_LIST,
)
from .platform_registry import PLATFORMS, detect_platform, get_names_map
from .handlers import HANDLER_MAP, handle_fallback
from .cache import agent_cache
from .stage_map import get_stage_label, get_stage_icon
from .registry import get as get_card, by_category, CATEGORY_DEPENDENCY_RULES
from .output_filter import content_quality_check
from .parser import parse_plan_with_reask
from .plan_validator import validate_plan, plan_to_dict, dict_to_plan
from .plan_frozen import FrozenPlan, UpgradeLevel

from src.core.exceptions import HandlerError, ToolExecutionError
from src.observability.metrics import record_agent_task, record_agent_run
from src.observability.agent_tracer import get_agent_tracer, AgentTracer
from src.logging_shared.agent_events import AgentEvent, TokenUsage, EventType
from src.logging_shared.tool_event_adapter import dispatch_tool_event
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


def _compute_ready_tasks(
    task_queue: list[dict], completed_ids: set, failed_ids: set
) -> list[dict]:
    """根据 task_queue 和已完成/失败集合计算当前可调度的任务。

    与 schedule_and_execute 内部的 deps_met 逻辑保持一致；
    给 should_continue_or_finish 用于在 ready_tasks 被清空时重算。
    """
    def deps_met(task: dict) -> bool:
        for dep in task.get("dependencies", []):
            dep_id = _resolve_dep_id(dep, task_queue)
            if dep_id is None:
                return False
            if dep_id not in completed_ids and dep_id not in failed_ids:
                return False
        return True

    return [
        t for t in task_queue
        if t.get("id") not in completed_ids
        and t.get("id") not in failed_ids
        and deps_met(t)
    ]


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


def _normalize_plan_for_context(plan: dict, context: dict, user_request: str) -> dict:
    """根据结构化上下文强制校正任务链，避免 revise/搜索分析回退到初稿链。"""
    tasks = list(plan.get("tasks", []))
    if not tasks:
        return plan

    action = str(context.get("action") or "").lower()
    revision_focus = context.get("revision_focus")
    existing_content = context.get("content") or context.get("draft_content") or ""

    if action == "revise":
        normalized: list[dict] = []
        review_task_id = None

        for task in tasks:
            task_copy = dict(task)
            task_input = dict(task_copy.get("input", {}))
            agent_type = task_copy.get("agent_type", "")

            if agent_type == "WritingAgent":
                continue

            if agent_type == "ReviewAgent":
                task_input.setdefault("topic", context.get("topic") or user_request)
                task_input["content"] = existing_content or task_input.get("content", "")
                if revision_focus:
                    task_input["revision_focus"] = revision_focus
                task_input["_rewrite"] = False
                task_copy["input"] = task_input
                task_copy["dependencies"] = []
                review_task_id = task_copy.get("id")
                normalized.append(task_copy)
                continue

            if agent_type == "FixAgent":
                task_input.setdefault("topic", context.get("topic") or user_request)
                task_input["content"] = existing_content or task_input.get("content", "")
                if revision_focus and not task_input.get("feedback"):
                    task_input["feedback"] = str(revision_focus)
                task_copy["input"] = task_input
                task_copy["dependencies"] = [review_task_id] if review_task_id else []
                normalized.append(task_copy)
                continue

            task_copy["input"] = task_input
            normalized.append(task_copy)

        if not any(t.get("agent_type") == "ReviewAgent" for t in normalized):
            review_task_id = str(uuid.uuid4())
            normalized.append({
                "id": review_task_id,
                "agent_type": "ReviewAgent",
                "input": {
                    "topic": context.get("topic") or user_request,
                    "content": existing_content,
                    "revision_focus": revision_focus or "",
                    "_rewrite": False,
                },
                "dependencies": [],
                "can_parallel_with": [],
                "retry_on_low_quality": False,
                "priority": "medium",
                "config": {"reasoning": "revise review", "priority": len(normalized)},
            })

        if not any(t.get("agent_type") == "FixAgent" for t in normalized):
            normalized.append({
                "id": str(uuid.uuid4()),
                "agent_type": "FixAgent",
                "input": {
                    "topic": context.get("topic") or user_request,
                    "content": existing_content,
                    "feedback": str(revision_focus or ""),
                },
                "dependencies": [review_task_id] if review_task_id else [],
                "can_parallel_with": [],
                "retry_on_low_quality": True,
                "priority": "medium",
                "config": {"reasoning": "revise fix", "priority": len(normalized)},
            })

        return {**plan, "tasks": normalized}

    # FIX #2: 旧 wants_search 关键词路径已删除
    # 原 P-25 逻辑由新的 plan_validator._inject_missing_deps 替代 (Layer 3)
    # 关键词匹配存在 over-injection 风险: "今天搜索" 也会触发不必要的 WebSearch
    # 统一交给 validator 基于 plan 拓扑 + 搜索信号 决定是否追加

    return {**plan, "tasks": tasks}


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
    if task_queue and task_id:
        try:
            if event_type == "thinking_step":
                await task_queue.publish_thinking_step(task_id, data)
            elif event_type == "llm_chunk":
                await task_queue.publish_llm_chunk(task_id, data)
            elif event_type == "reasoning":
                await task_queue.publish_reasoning(task_id, data)
            elif event_type == "stage":
                await task_queue.publish_stage(task_id, data)
            elif event_type == "error":
                # FIX D: 节点内部异常也能即时推 error 到 SSE
                task_queue.fail_task(task_id, data.get("message", "节点失败"))
            elif event_type == "result":
                # FIX: 节点出 content 也能走此 path (较少用, 但兜底)
                task_queue.complete_task(task_id, data)
        except Exception as e:
            logger.warning("发布流式事件失败 (event_type=%s): %s", event_type, e)


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
        logger.warning("后处理上下文失败: %s (agent_type=%s)", e, agent_type)


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
        # ── 在 LLM 调用前发布 stage:started，让前端立即看到思考分析阶段 ──
        await _publish_streaming_event(state, "stage", {
            "agent": "ManagerAgent",
            "label": "思考分析中",
            "icon": "🧠",
            "status": "started",
            "timestamp": time.time(),
        })

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
                logger.warning("LLM 理解失败，使用规则兜底: %s", str(e)[:200])
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

        # Emit reasoning 事件（发送 LLM 理解分析的完整结果）
        intent_info = understanding.get("intent", {})
        slots = understanding.get("slots", {})
        platform_name = understanding.get("platform_name", "")
        safety = understanding.get("safety", True)
        reasoning_parts = [
            f"📋 意图分析：{intent_info.get('coarse', '')} - {intent_info.get('fine', '')}",
        ]
        if platform_name:
            reasoning_parts.append(f"📍 目标平台：{platform_name}")
        if understanding.get("cleaned_prompt"):
            reasoning_parts.append(f"💬 核心需求：{understanding['cleaned_prompt'][:100]}")
        if slots.get("tone"):
            reasoning_parts.append(f"🎨 写作风格：{slots['tone']}")
        if slots.get("length_hint"):
            reasoning_parts.append(f"📐 篇幅要求：{slots['length_hint']}")
        if slots.get("audience"):
            reasoning_parts.append(f"👥 目标读者：{slots['audience']}")
        if understanding.get("language"):
            reasoning_parts.append(f"🌐 语言：{understanding['language']}")
        reasoning_parts.append(f"🔒 安全检查：{'✅ 通过' if safety else '❌ 未通过'}")
        _reasoning_text = "\n".join(reasoning_parts)
        await _publish_streaming_event(state, "reasoning", {
            "agent": "ManagerAgent",
            "content": _reasoning_text,
            "timestamp": time.time(),
        })

        await _publish_streaming_event(state, "stage", {
            "agent": "ManagerAgent",
            "status": "completed",
            "duration": time.time() - (state.get("start_time", time.time()) or time.time()),
            "timestamp": time.time(),
            "_reasoning": _reasoning_text,  # 携带推理内容到 stage 事件
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
        # 仍然补一个 stage:completed 让前端 timeline 一致（理解阶段完成后立刻显示执行路线）
        # 注意：使用独立的 "PlannerAgent" key，避免与 understand_intent 的 "ManagerAgent" 冲突
        await _publish_streaming_event(state, "stage", {
            "agent": "PlannerAgent",
            "label": "规划任务中",
            "icon": "📋",
            "status": "completed",
            "duration": 0,
            "timestamp": time.time(),
            "_reasoning": "使用预计算计划",
        })
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

        # ── 在 LLM 调用前发布 stage:started ──
        # 注意：使用独立的 "PlannerAgent" key，避免与 understand_intent 的 "ManagerAgent" 冲突，
        #     这样前端 timeline 能看到"思考分析中" → "规划任务中" 两个独立阶段
        await _publish_streaming_event(state, "stage", {
            "agent": "PlannerAgent",
            "label": "规划任务中",
            "icon": "📋",
            "status": "started",
            "timestamp": time.time(),
        })

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
                # ── Layer 2: Pydantic + reask parse (替换旧的 regex/json.loads 路径) ──
                                # 读取上一次校验的 issues（若存在，说明是 replan）
                _prev_issues = state.get("_validate_issues", [])
                _issues_section = ""
                if _prev_issues:
                    _issues_text = "\n".join(f"- {e}" for e in _prev_issues)
                    _issues_section = (
                        "\n## 上一次规划的问题（请修正）\n"
                        + _issues_text
                        + "\n"
                    )

                user_planner_prompt = f"""## User Request
{cleaned_prompt}

## Intent Understanding
{context_str}
{_issues_section}
Based on the above understanding, select the most appropriate agents to complete the task.
Output ONLY the JSON plan, no extra text."""
                parsed_plan = await parse_plan_with_reask(
                    llm, user_planner_prompt,
                    context_str=context_str,
                    max_attempts=3,
                    timeout=60.0,
                    available_agents_desc=_PLANNING_AGENTS_LIST,
                )

                if parsed_plan is None:
                    logger.warning(
                        "LLM 计划 parse+reask 全部失败, 使用兜底方案. "
                        "LLM响应: %s", (response.content[:300] if hasattr(response, 'content') and response.content else "空")
                    )
                    plan = _build_fallback_plan(cleaned_prompt, detected_platform)
                else:
                    # parse_plan_with_reask 已完成"解析 + 校验 + reask"，
                    # 返回的 plan 必然通过所有校验，无需重复 validate
                    plan = plan_to_dict(parsed_plan)
                    plan["plan_method"] = "llm"
                    logger.info(
                        "LLM 计划已生成：%d 个任务",
                        len(plan.get("tasks", [])),
                    )
                    _tasks = plan.get("tasks", [])
                    _steps = []
                    for _t in _tasks:
                        _deps = _t.get("dependencies", [])
                        _dep_str = f"[dep:{_deps[0]}]" if _deps else ""
                        _steps.append(f"  {_t.get('agent_type','?')} {_dep_str}")
                    logger.info("▸ 执行路线 ─────────────────────\n%s\n──────────────────────────", "\n".join(_steps))

            except Exception as e:
                logger.warning("LLM 规划失败，使用兜底方案: %s", str(e)[:200])
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

        # 注入前端传入的结构化参数（覆盖 slots，优先级更高）
        ctx = state.get("context", {})
        _PARAM_KEYS = [
            "topic", "platform", "style", "tone", "audience", "keywords", "length",
            "action", "revision_focus", "content", "draft_content",
            "content_type",   # WritingAgent 的内容形式参数（如 blog/xhs_post/copywriting）
        ]
        for task in plan.get("tasks", []):
            task_input = task.get("input", {})
            for key in _PARAM_KEYS:
                val = ctx.get(key)
                if val is not None and val not in ("", "null", None):
                    task_input[key] = val
            # 把整个 context 透传给 handler（供 _agent_context 使用）
            task_input["_agent_context"] = {k: ctx.get(k) for k in _PARAM_KEYS if ctx.get(k)}

        plan = _normalize_plan_for_context(plan, ctx, cleaned_prompt)

        # 自动修正依赖链（防止 LLM 漏设依赖导致并行错误）
        # 从 registry 的 CATEGORY_DEPENDENCY_RULES 推断
        tasks = plan.get("tasks", [])

        # 按类别分组 task_id（用于依赖注入）
        _category_task_ids: dict[str, list[str]] = {}
        for t in tasks:
            card = get_card(t.get("agent_type", ""))
            cat = card.category if card else ""
            _category_task_ids.setdefault(cat, []).append(t["id"])

        # 自动修正依赖链 — auto-fixer 已在 plan_validator 中执行，
        # 但它注入的任务可能缺少被依赖方的反向引用。
        # 例如：auto-fixer 注入了 WebSearch，但 WritingAgent 需要依赖 WebSearch。
        # 这里补上缺失的依赖，但过滤掉 auto-fixer 刚注入的任务（避免循环依赖）。
        _auto_ids: set[str] = set()
        # 从 plan 中找出 id 以 "auto-" 开头的（auto-fixer 注入的任务）
        for t in tasks:
            tid = t.get("id", "")
            if tid.startswith("auto-"):
                _auto_ids.add(tid)

        # "search" 类任务（WebSearch/URLCrawler）始终为根任务，不依赖任何人
        # 防止 LLM 误设 WebSearch 依赖 SearchAnalyzer 等反向依赖
        for t in tasks:
            card = get_card(t.get("agent_type", ""))
            if card and card.category == "search":
                t["dependencies"] = []

        for t in tasks:
            card = get_card(t.get("agent_type", ""))
            if not card:
                continue
            deps = t.setdefault("dependencies", [])
            dep_cats = CATEGORY_DEPENDENCY_RULES.get(card.category, [])
            for dep_cat in dep_cats:
                dep_ids = _category_task_ids.get(dep_cat, [])
                # 过滤掉 auto-fixer 刚注入的 id，避免循环依赖
                filtered = [did for did in dep_ids if did not in _auto_ids]
                if filtered and filtered[-1] not in deps:
                    deps.append(filtered[-1])

        # 清理 task 依赖：剔除自引用（task 写自己为 dep）和循环引用
        # LLM 偶尔会误把"dep: 0"写到 task[1] 上，导致整条链死锁
        _tasks_raw = plan.get("tasks", [])
        for _i, _t in enumerate(_tasks_raw):
            _deps = _t.get("dependencies", [])
            if not _deps:
                continue
            _clean_deps = []
            for _d in _deps:
                # 自引用：dep 数字索引等于自己 index，或 dep 字符串等于自己 id
                if isinstance(_d, int) and _d == _i:
                    continue
                if isinstance(_d, str) and _d == _t.get("id"):
                    continue
                _clean_deps.append(_d)
            if _clean_deps != _deps:
                logger.info("清理 task[%d] %s 的自引用依赖: %s -> %s",
                            _i, _t.get("agent_type", "?"), _deps, _clean_deps)
                _t["dependencies"] = _clean_deps

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

        # Emit reasoning 事件展示任务拆解结果
        tasks = plan.get("tasks", [])
        reasoning_lines = [f"📋 执行计划：共 {len(tasks)} 个任务"]
        for i, t in enumerate(tasks):
            agent_type = t.get("agent_type", "?")
            deps = t.get("dependencies", [])
            dep_str = f" [依赖: {', '.join(str(d) for d in deps[:3])}]" if deps else ""
            reasoning_lines.append(f"  {i+1}. {agent_type}{dep_str}")
        _plan_reasoning = "\n".join(reasoning_lines)
        await _publish_streaming_event(state, "reasoning", {
            "agent": "ManagerAgent",
            "content": _plan_reasoning,
            "timestamp": time.time(),
        })

        span.add_event("plan_built", {
            "task_count": len(plan.get("tasks", [])),
            "method": plan.get("plan_method", "fallback"),
        })
        span.set_attribute("plan.method", plan.get("plan_method", "fallback"))
        span.set_attribute("agent.task_count", len(plan.get("tasks", [])))
        span.set_status("OK")

        # 规划完成事件（started 已在 LLM 调用前发出）
        # 使用 "PlannerAgent" key 与上面的 started 配对
        await _publish_streaming_event(state, "stage", {
            "agent": "PlannerAgent",
            "status": "completed",
            "duration": time.time() - (state.get("start_time", time.time()) or time.time()),
            "timestamp": time.time(),
            "_reasoning": _plan_reasoning,
        })

        # ── Layer 4: 冻结 plan (retry→patch→replan 升级协议的状态基础) ──
        try:
            pyd_plan = dict_to_plan(plan)
            frozen = FrozenPlan.freeze(pyd_plan)
            updates["frozen_plan"] = frozen.to_dict()
            updates["plan_version"] = frozen.version_hash
        except Exception as _e:
            # 冻结失败不影响 plan 本身使用, 仅记录
            logger.warning("plan freeze failed (non-fatal)", error=str(_e)[:200])
            updates["frozen_plan"] = None
            updates["plan_version"] = None

        # plan_tasks 成功生成 plan 时设置 plan_valid=True
        # 这样 _after_validate 条件边能正确路由到 execute
        updates["plan_valid"] = True

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

    logger.info(
        "schedule_and_execute: total=%d completed=%d",
        len(task_queue), len(completed_ids),
    )

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

    # Debug: 列出每个 task 的 deps 解析情况
    for t in task_queue:
        if t.get("id") in completed_ids or t.get("id") in failed_ids:
            continue
        _resolved = [_resolve_dep_id(d, task_queue) for d in t.get("dependencies", [])]
        logger.info(
            "  task %s deps=%s resolved=%s ready=%s",
            t.get("agent_type", "?"),
            t.get("dependencies", []),
            _resolved,
            t in ready,
        )

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
            # 死锁检测：所有 stuck 任务的 dep 指向的都是 stuck 任务本身（循环依赖）
            # 或 stuck 任务有 deps 但无任何 deps 被满足（真死锁，非正常等待）
            _stuck_ids = {t.get("id") for t in stuck}
            _cycle = all(
                _resolve_dep_id(d, task_queue) in _stuck_ids
                for t in stuck for d in t.get("dependencies", [])
            ) if stuck and any(t.get("dependencies") for t in stuck) else False
            if _cycle or deps_unmet == 0:
                logger.warning("检测到死锁 (stuck=%d, deps_unmet=%d)", len(stuck), deps_unmet)
                span.add_event("deadlock_detected", {"stuck_count": len(stuck)})
                span.set_status("ERROR")
                # 死锁：打断循环依赖，让第一个 stuck task 无视依赖执行
                first_stuck = stuck[0]
                logger.warning(
                    "死锁打断: 清除 %s 依赖",
                    first_stuck.get("agent_type", "?"),
                )
                # 从 task_queue 中清除该 task 的依赖（持久化到 state）
                for t in task_queue:
                    if t.get("id") == first_stuck.get("id"):
                        t["dependencies"] = []
                        break
                return {
                    "ready_tasks": [first_stuck],
                    "task_queue": task_queue,
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
        # FixAgent 循环：最多 _MAX_REVIEW_LOOPS 轮，不提前跳过（让 FixAgent 跑满次数尽力优化）
        _skipped_tasks: list[dict] = []

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
                        # 写入共享结果 dict（同一批次内的其他任务可见）
                        state.setdefault("_shared_results", {})[task_id] = result
                        return task_id, result
                    except Exception as e:
                        task_span.set_status("ERROR")
                        task_span.record_exception(e)
                        task_span.set_attribute("task.status", "failed")
                        raise

        tasks_with_steps = [(t, base_step + i + 1) for i, t in enumerate(ready_tasks)]
        batch_span.add_event("batch_started", {"task_count": len(tasks_with_steps)})
        # 日志：当前批次进度
        _batch_agents = [t.get("agent_type","?") for t, _ in tasks_with_steps]
        _done = len(state.get("completed_task_ids", []))
        _total = len(state.get("task_queue", []))
        logger.info("▸ 批次 %d/%d → %s", _done, _total, " → ".join(_batch_agents))

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
                logger.error("Agent 任务抛出异常 (task_id=%s)", task_id, exc_info=True)
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

            # Phase 0 关键改动：WritingAgent 完成 → dispatch handoff (Writing→Review)
            agent_type_done = result.get("agent_type", "")
            if agent_type_done == "WritingAgent" and result.get("status") in ("completed", "partial"):
                try:
                    from src.logging_shared.genai_records import HandoffRecord
                    from src.logging_shared.genai_dispatcher import dispatch_handoff
                    _writing_output = result.get("output", {}).get("content", "")[:500] if isinstance(result.get("output"), dict) else ""
                    await dispatch_handoff(HandoffRecord(
                        from_agent="WritingAgent",
                        to_agent="ReviewAgent",
                        handoff_reason="task_completed",
                        trace_id=state.get("trace_task_id", "") or "",
                        span_id=task_id,
                        parent_span_id=None,
                        task_id=task_id,
                        state_snapshot={
                            "writing_output_excerpt": _writing_output,
                            "tokens_used": result.get("tokens_used", 0),
                        },
                    ))
                except Exception:
                    pass

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
            # 发送 stage:completed 事件通知前端更新（否则前端 FixAgent 一直卡在"优化中"）
            review_round = st.get("input", {}).get("_review_round", 0)
            fix_completed = {
                "agent": "FixAgent",
                "status": "completed",
                "duration": 0,
                "timestamp": time.time(),
            }
            if review_round > 0:
                fix_completed["_review_round"] = review_round
            await _publish_streaming_event(state, "stage", fix_completed)

        # 检查本轮完成的 ReviewAgent 是否通过了审核
        # 用于短路后续 review-fix 循环，避免质量已达标时继续浪费 Token
        # 注意：只有 re-review 轮次（_review_round > 0）的通过才算数，
        # 初审（round 0）的格式可靠性不如复审（LLM 可能遗漏结论行），
        # 如果让它阻塞循环，FixAgent 完成后就不会追加复审+再润色。
        _last_review_passed = state.get("_last_review_passed", False)
        task_queue = state.get("task_queue", [])
        _skip_fixagent_ids: list[str] = []
        for task_id, result in list(task_results.items()):
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
                # 收集要跳过的 FixAgent，稍后统一处理（避免迭代时修改 task_results）
                for t in task_queue:
                    if t.get("agent_type") == "FixAgent" and task_id in t.get("dependencies", []):
                        _skip_fixagent_ids.append(t["id"])
            elif isinstance(output, dict) and output.get("passed") is False:
                # Phase 0 关键改动：ReviewAgent 判定不通过 → dispatch handoff
                # 事故复盘时这是关键证据：为什么 ReviewAgent 决定 fix
                try:
                    from src.logging_shared.genai_records import HandoffRecord
                    from src.logging_shared.genai_dispatcher import dispatch_handoff
                    _parent_review_state = state.get("_review_feedback_for_fix", "") or ""
                    await dispatch_handoff(HandoffRecord(
                        from_agent="ReviewAgent",
                        to_agent="FixAgent",
                        handoff_reason=f"review_failed_round_{review_round}",
                        trace_id=state.get("trace_task_id", "") or "",
                        span_id=task_id,
                        parent_span_id=None,
                        task_id=task_id,
                        state_snapshot={
                            "review_round": review_round,
                            "review_passed": False,
                            "review_feedback": _parent_review_state[:500],
                            "review_output": str(output)[:500],
                        },
                    ))
                except Exception:
                    pass

        for skip_id in _skip_fixagent_ids:
            if skip_id not in task_results:
                task_results[skip_id] = {
                    "status": "skipped", "agent_type": "FixAgent",
                    "output": {"content": "", "passed": True, "skipped": True},
                }
            if skip_id not in completed:
                completed.append(skip_id)
            # 同时找对应的 review_round 通知前端
            t = next((t for t in task_queue if t.get("id") == skip_id), None)
            await _publish_streaming_event(state, "stage", {
                "agent": "FixAgent",
                "status": "completed",
                "duration": 0,
                "timestamp": time.time(),
                "_review_round": t.get("input", {}).get("_review_round", 0) if t else 0,
            })

        # ── Review-Fix 循环检测（最多 2 轮，每批只允许生成一轮，防止指数爆炸） ──
        _MAX_REVIEW_LOOPS = 2
        _disable_review_loop = str(state.get("context", {}).get("action") or "").lower() == "revise"

        # 从 task_queue 中计算当前 review-fix 轮数，不依赖 state 累积
        # （因为 LangGraph 状态合并可能因 TypedDict 列表/字典 reducer 问题导致
        #  _review_fix_cycles 不准，表现为日志始终显示"批次 0/4"）
        _existing_review_rounds = set()
        for t in state.get("task_queue", []):
            rr = t.get("input", {}).get("_review_round", 0) if isinstance(t.get("input"), dict) else 0
            if rr > 0:
                _existing_review_rounds.add(rr)
        _cycle_from_queue = max(_existing_review_rounds) if _existing_review_rounds else 0

        # 用更可靠的队列推断值覆盖 state 值
        _review_fix_cycles = _cycle_from_queue
        _loop_already_created = state.get("_loop_created_in_cycle", False)

        loop_new_tasks: list[dict] = []
        if (
            not _disable_review_loop
            and _review_fix_cycles < _MAX_REVIEW_LOOPS
            and not _loop_already_created
            and not _last_review_passed
        ):
            # 找本轮刚完成的最新 FixAgent（按 task_queue 顺序取最后一个）
            fix_candidates = [
                tid for tid in completed
                if task_results.get(tid, {}).get("agent_type") == "FixAgent"
                and any(t.get("id") == tid for t in state.get("task_queue", []))
            ]
            if fix_candidates:
                tid = fix_candidates[-1]  # 取最后一个（最新完成的）
                result = task_results.get(tid, {})
                fix_task = next((t for t in state.get("task_queue", []) if t.get("id") == tid), None)

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

        # ── 提取最新 ReviewAgent 审核意见到专用通道（避免 task_results 状态累积问题） ──
        _review_feedback_for_fix = ""
        for _r in task_results.values():
            if _r.get("agent_type") == "ReviewAgent" and _r.get("status") in ("completed", "partial"):
                _out = _r.get("output", {})
                _ct = _out.get("content", "") if isinstance(_out, dict) else ""
                if _ct:
                    _review_feedback_for_fix = _ct  # 取最后一个（最新的）

        shared_results = dict(state.get("_shared_results", {}))
        shared_results.update(task_results)

        updates = {
            "task_results": task_results,
            "completed_task_ids": completed,
            "failed_task_ids": failed,
            "ready_tasks": [],
            "thinking_steps": existing + new_steps,
            "_last_review_passed": _last_review_passed,
            "_shared_results": shared_results,
            "_review_feedback_for_fix": _review_feedback_for_fix,
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

    # Phase 0 关键改动：每次进入子 Agent 时 push 一个 trace 上下文，
    # 让 handler 内 call_llm 自动关联 trace_id / span_id。
    # 根 trace_id 来自 HTTP 请求（_trace_task_id）；span_id 是本次 Agent 的独立 span。
    from src.logging_shared.trace_context_var import (
        push_trace, pop_trace, new_span_id,
    )
    _task_trace_id = state.get("trace_task_id", "") or ""
    _agent_trace_token = push_trace(
        trace_id=_task_trace_id or None,
        span_id=new_span_id(),
        parent_span_id=None,
        agent_name=agent_type,
    )
    _active_trace_id = _agent_trace_token.trace_id
    _active_span_id = _agent_trace_token.span_id

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
            user_id = state.get("context", {}).get("user_id", "")
            t_input["_task_id"] = task.get("id") or task_id or agent_type
            t_input["_session_id"] = task_session_id
            t_input["session_id"] = task_session_id
            if user_id:
                t_input["user_id"] = user_id

            # 上下文注入
            dep_context = _build_dependency_context(task, state)
            if dep_context:
                t_input["_context"] = dep_context

            # ── 通用下游 content/feedback 兜底 ──
            # 让 ReviewAgent / SEOAgent 等所有需要上游正文的下游 agent
            # 都能从 _context 里拿到第一个非空 content；feedback 从 ReviewAgent 拿
            # 注意：FixAgent 有专用的 multi-tier 注入（包含 content+feedback 精准路由），
            # 使用通用兜底会误将 ReviewAgent 输出（审核意见）设为其 content。
            if agent_type != "FixAgent":
                if not t_input.get("content"):
                    for _cdata in dep_context.values():
                        _c_cont = _cdata.get("content", "")
                        if _c_cont:
                            t_input["content"] = _c_cont
                            break
            if not t_input.get("feedback"):
                for _cdata in dep_context.values():
                    if _cdata.get("agent") == "ReviewAgent":
                        t_input["feedback"] = _cdata.get("content", "")
                        break

            # Skill context injection — 仅流向需要创作指导的 WritingAgent
            # 其它 agent（Review/Code/Diagram 等）有各自独立的系统 prompt，
            # 不应接收 skill 带来的"角色定位"信息（会导致审核者变成写作者等问题）
            skill_system_prompt = state.get("skill_system_prompt", "")
            if skill_system_prompt and agent_type == "WritingAgent":
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

            # 记忆检索工具（使 agent 能主动查询记忆）
            _MEMORY_TOOL_DESC = """
## 记忆检索工具
需要参考历史内容或用户偏好时，可以使用以下工具。Handler 会自动执行：

- memory_search(query: str) — 搜索相关历史记忆
  例：memory_search("甜品教程 风格偏好")

- user_preferences() — 获取当前用户的写作偏好（平台、风格、语调等）

- recent_generations(limit: 5) — 查询最近生成的内容记录
"""
            existing_tools = t_input.get("_tool_descriptions", "")
            if _MEMORY_TOOL_DESC not in existing_tools:
                t_input["_tool_descriptions"] = existing_tools + _MEMORY_TOOL_DESC
            # 将记忆工具函数注入 t_input，供 handler 调用
            from src.memory.tools import run_memory_tool, MEMORY_TOOLS
            t_input["_memory_tools_run"] = run_memory_tool
            t_input["_memory_tools_list"] = MEMORY_TOOLS

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
                    cache_completed = {
                        "agent": agent_type,
                        "status": "completed",
                        "duration": elapsed,
                        "timestamp": time.time(),
                    }
                    if review_round > 0:
                        cache_completed["_review_round"] = review_round
                    await _publish_streaming_event(state, "stage", cache_completed)
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
                        logger.warning("流式事件发布失败 (event_type=%s): %s", event_type, e)

                # Agent → 内容通道映射，前端据此路由渲染（不再靠 agent 名称推测）
                _CHANNEL_MAP: dict[str, str] = {
                    "WritingAgent": "body",
                    "FixAgent": "body",
                    "ReviewAgent": "review",
                    "JudgeAgent": "review",
                    "CriticAgent": "review",
                    "WebSearch": "search",
                    "URLCrawler": "search",
                    "SearchAnalyzer": "search",
                    "SEOAgent": "body",
                    "CodeAgent": "body",
                    "DiagramAgent": "body",
                    "AnalysisAgent": "review",
                }
                _agent_channel = _CHANNEL_MAP.get(agent_type, "body")

                def _make_on_llm_chunk(agent: str, rr: int = 0, ch: str = "body"):
                    async def _on_llm_chunk(chunk: str) -> None:
                        data = {"agent": agent, "content": chunk, "channel": ch, "timestamp": time.time()}
                        if rr > 0:
                            data["_review_round"] = rr
                        await _safe_publish("llm_chunk", data)
                    return _on_llm_chunk
                def _make_on_reasoning(agent: str, rr: int = 0):
                    async def _on_reasoning(text: str) -> None:
                        data = {"agent": agent, "content": text, "timestamp": time.time()}
                        if rr > 0:
                            data["_review_round"] = rr
                        await _safe_publish("reasoning", data)
                        # 同时写 ES
                        try:
                            await emitter.emit(AgentEvent(
                                event_type="thought",
                                agent_name=agent,
                                thought=text[:2000],
                                run_id=task_id or agent,
                                llm_model=getattr(llm, 'model_name', '') if llm else '',
                            ))
                        except Exception:
                            pass
                    return _on_reasoning
                t_input["_on_llm_chunk"] = _make_on_llm_chunk(agent_type, review_round, _agent_channel)
                t_input["_on_reasoning"] = _make_on_reasoning(agent_type, review_round)

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

            # Emit tool_call 事件到 ES（记录本次 agent 要做什么）
            _task_desc = str(task_input.get("topic", task_input.get("query", task_input.get("task", ""))))[:300]
            await emitter.emit(AgentEvent(
                event_type="tool_call",
                agent_name=agent_type,
                tool_name="handler_execute",
                tool_input={"task": _task_desc, "input_keys": list(t_input.keys())[:10]},
                run_id=task_id or agent_type,
                llm_model=getattr(llm, 'model_name', '') if llm else '',
            ))
            # 同步 dispatch_tool_event 到 agent-events-* (event_type=tool_call)
            try:
                await dispatch_tool_event(
                    event_type="tool_call",
                    tool_name="handler_execute",
                    params={"task": _task_desc, "input_keys": list(t_input.keys())[:10], "agent_type": agent_type},
                    task_id=task.get("id") or task_id or None,
                )
            except Exception:
                pass

            # 被动注入记忆上下文（handler 主动查不到时兜底）
            _uid = state.get("context", {}).get("user_id", "")
            if _uid and not t_input.get("_memory_context"):
                try:
                    from src.memory import memory_manager
                    _ctx = await memory_manager.build_agent_context(
                        agent_type=agent_type, task_input=t_input,
                        user_id=_uid,
                        conversation_id=state.get("context", {}).get("conversation_id", ""),
                    )
                    if _ctx:
                        t_input["_memory_context"] = _ctx
                except Exception as e:
                    logger.warning("被动记忆注入失败: %s", e)

            # FixAgent：专用通道注入 + 多路兜底
            if agent_type == "FixAgent":
                _tq = state.get("task_queue", [])
                _dep_ctx = t_input.get("_context", {})

                # ── 第一优先级：专用反馈通道 _review_feedback_for_fix（由 execute_task_batch 维护，最可靠） ──
                if not t_input.get("feedback"):
                    _feedback = state.get("_review_feedback_for_fix", "")
                    if _feedback:
                        t_input["feedback"] = _feedback

                # ── 第二优先级：从已构建的 _context 提取（_build_dependency_context 已执） ──
                if not t_input.get("content") or not t_input.get("feedback"):
                    for _cdata in _dep_ctx.values():
                        _c_agent = _cdata.get("agent", "")
                        _c_cont = _cdata.get("content", "")
                        if not _c_cont:
                            continue
                        if _c_agent in ("WritingAgent", "FixAgent") and _c_cont:
                            t_input["content"] = _c_cont
                        elif _c_agent == "ReviewAgent" and not t_input.get("feedback"):
                            t_input["feedback"] = _c_cont

                # ── 第三优先级：全量扫描 task_results + _shared_results ──
                _all_fix = {}
                try:
                    _all_fix.update(state.get("task_results", {}))
                except Exception:
                    pass
                try:
                    _all_fix.update(state.get("_shared_results", {}))
                except Exception:
                    pass
                if not t_input.get("content") or not t_input.get("feedback"):
                    for _r in _all_fix.values():
                        _at = _r.get("agent_type", "")
                        _out = _r.get("output", {})
                        _ct = (_out.get("content", "") if isinstance(_out, dict)
                               else str(_out) if _out else "")
                        if _at in ("WritingAgent", "FixAgent") and _ct:
                            t_input["content"] = _ct
                        elif _at == "ReviewAgent" and _ct and not t_input.get("feedback"):
                            t_input["feedback"] = _ct

                # ── 第四优先级：兜底用 topic ──
                if not t_input.get("content"):
                    t_input["content"] = task.get("input", {}).get("topic", "")
                if not t_input.get("feedback"):
                    t_input["feedback"] = ""
                _dep_ctx = t_input.get("_context", {})

                # ── 第 1 层：从已构建的 _context 提权（最可靠，_build_dependency_context 已执行） ──
                if not t_input.get("content") or not t_input.get("feedback"):
                    for _cdata in _dep_ctx.values():
                        _c_agent = _cdata.get("agent", "")
                        _c_cont = _cdata.get("content", "")
                        if not _c_cont:
                            continue
                        if _c_agent in ("WritingAgent", "FixAgent") and _c_cont:
                            t_input["content"] = _c_cont
                        elif _c_agent == "ReviewAgent" and not t_input.get("feedback"):
                            t_input["feedback"] = _c_cont

                # ── 第 2 层：全量扫描 task_results + _shared_results ──
                _all_fix = {}
                try:
                    _all_fix.update(state.get("task_results", {}))
                except Exception:
                    pass
                try:
                    _all_fix.update(state.get("_shared_results", {}))
                except Exception:
                    pass
                if not t_input.get("content") or not t_input.get("feedback"):
                    for _r in _all_fix.values():
                        _at = _r.get("agent_type", "")
                        _out = _r.get("output", {})
                        _ct = (_out.get("content", "") if isinstance(_out, dict)
                               else str(_out) if _out else "")
                        if _at in ("WritingAgent", "FixAgent") and _ct:
                            t_input["content"] = _ct
                        elif _at == "ReviewAgent" and _ct and not t_input.get("feedback"):
                            t_input["feedback"] = _ct

                # ── 第 3 层：遍历 task_queue 找到 FixAgent 依赖链中 ReviewAgent ──
                if not t_input.get("content") or not t_input.get("feedback"):
                    for _dep_ref in task.get("dependencies", []):
                        _dep_id = _resolve_dep_id(_dep_ref, _tq)
                        if not _dep_id:
                            continue
                        # 检查依赖的 agent_type 是否为 ReviewAgent
                        _dep_task = next((_t for _t in _tq if _t.get("id") == _dep_id), None)
                        if not _dep_task:
                            continue
                        _dep_card = get_card(_dep_task.get("agent_type", ""))
                        if not _dep_card or _dep_card.category != "review":
                            continue
                        # 找到 ReviewAgent 的结果
                        _r_result = _all_fix.get(_dep_id) or {}
                        _r_out = _r_result.get("output", {})
                        _r_ct = (_r_out.get("content", "") if isinstance(_r_out, dict)
                                 else str(_r_out) if _r_out else "")
                        if _r_ct and not t_input.get("feedback"):
                            t_input["feedback"] = _r_ct

                # ── 第 4 层：死马当活马医 — 直接用当前 task 的 task_input 原始内容 ──
                if not t_input.get("content"):
                    t_input["content"] = task.get("input", {}).get("topic", "")
                if not t_input.get("feedback"):
                    t_input["feedback"] = ""  # 空 feedback，handler 会保留原文

            result = await _safe_run_handler(agent_type, handler, t_input, llm, _mcp, task=task, state=state)

            # 归一化为标准信封格式（兼容旧 handler）
            result = _normalize_agent_result(result, agent_type)

            # 清理不可序列化的回调引用
            t_input.pop("_on_llm_chunk", None)
            t_input.pop("_on_reasoning", None)

            elapsed = round(time.time() - start_time, 2)
            result["time_elapsed"] = elapsed

            # Emit stage:completed 事件
            review_round = task_input.get("_review_round", 0)
            stage_completed = {
                "agent": agent_type,
                "status": "completed",
                "duration": elapsed,
                "timestamp": time.time(),
            }
            if review_round > 0:
                stage_completed["_review_round"] = review_round
            await _publish_streaming_event(state, "stage", stage_completed)

            # Emit tool_result 事件到 ES
            _output_content = result.get("output", {})
            if isinstance(_output_content, dict):
                _output_text = _output_content.get("content", "")
            else:
                _output_text = str(_output_content)[:300]
            tokens = result.get("tokens_used", 0)
            if isinstance(tokens, dict):
                tokens = sum(tokens.values()) if tokens else 0
            await emitter.emit(AgentEvent(
                event_type="tool_result",
                agent_name=agent_type,
                tool_output=str(_output_text)[:500] if _output_text else (result.get("error", "") or "完成")[:500],
                message=result.get("status", "completed"),
                run_id=task_id or agent_type,
                duration_ms=elapsed * 1000,
                llm_model=getattr(llm, 'model_name', '') if llm else '',
                token_usage=TokenUsage(total_tokens=int(tokens)) if tokens else None,
            ))
            # Phase 0 关键改动：tool_result 同步 dispatch_tool_event 到独立索引
            try:
                await dispatch_tool_event(
                    event_type="tool_result",
                    tool_name="handler_execute",
                    output_summary=str(_output_text)[:500] if _output_text else (result.get("error", "") or "完成")[:500],
                    duration_ms=elapsed * 1000,
                    task_id=task.get("id") or task_id or None,
                )
            except Exception:
                pass

            # 沉淀记忆（仅对内容产出型 Agent）
            if _uid and agent_type in ("WritingAgent", "WebSearch", "SEOAgent", "DiagramAgent"):
                try:
                    from src.memory import memory_manager
                    await memory_manager.remember(
                        agent_type=agent_type,
                        task_input=t_input,
                        result=result,
                        user_id=_uid,
                        conversation_id=state.get("context", {}).get("conversation_id", ""),
                    )
                except Exception as e:
                    logger.warning("记忆沉淀失败: %s", e)

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
                    logger.warning("缓存结果失败 (agent_type=%s): %s", agent_type, e)

            return result

        # handler 内的异常已由 _safe_run_handler 兜底，这里的 except
        # 只兜底归一化/日志/缓存等编排代码的意外错误
        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            import traceback
            traceback.print_exc()
            logger.error(
                "Agent 编排异常 (agent_type=%s): %s",
                agent_type, e,
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

            # Phase 0 关键改动：编排异常也 dispatch handoff 记录
            try:
                from src.logging_shared.genai_records import HandoffRecord
                from src.logging_shared.genai_dispatcher import dispatch_handoff
                await dispatch_handoff(HandoffRecord(
                    from_agent=f"({agent_type})",
                    to_agent="(orchestration_error)",
                    handoff_reason="orchestration_exception",
                    trace_id=_active_trace_id,
                    span_id=_active_span_id,
                    parent_span_id=None,
                    task_id=task.get("id") or task_id or None,
                    state_snapshot={
                        "error_type": type(e).__name__,
                        "error_message": str(e)[:500],
                        "agent_type": agent_type,
                    },
                ))
            except Exception:
                pass

            from .result import error_response
            return error_response(
                agent_type,
                error=f"编排异常: {e}",
                failure_type="orchestration",
                error_type=type(e).__name__,
                time_elapsed=elapsed,
            )
        finally:
            # Phase 0 关键改动：清理子 Agent 的 trace 上下文
            try:
                pop_trace(_agent_trace_token)
            except Exception:
                pass

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
            "MCP 工具执行失败 (agent_type=%s, tool_name=%s): %s",
            agent_type, getattr(e, 'tool_name', ''), e,
            _type="agent", event_name="tool_execution_error",
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
            "Handler 执行失败 (agent_type=%s, handler=%s): %s",
            agent_type, getattr(e, 'handler', ''), e,
            _type="agent", event_name="handler_error",
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
            "Agent handler 异常 (agent_type=%s): %s",
            agent_type, e, exc_info=True,
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

    传递完整信封，包括 type、tokens_used、metadata 等结构化信息。
    每个依赖的内容截断到 budget 以内。

    优先级：_shared_results（当前批次并行任务成果） > task_results（已完成任务）
    同一 dep_id 在两个来源都存在时，优先取共享结果（最新）。
    """
    task_queue = state.get("task_queue", [])
    task_results = state.get("task_results", {})
    shared_results = state.get("_shared_results", {})
    context = {}
    dep_count = len(task.get("dependencies", []))
    budget_per_dep = max(1000, 6000 // max(dep_count, 1))

    for dep in task.get("dependencies", []):
        dep_id = _resolve_dep_id(dep, task_queue)
        if dep_id is None:
            continue
        # 读取顺序：task_results（已提交的历史批次）→ _shared_results（当前批次同行任务）
        # 注意：task_results 优先于 _shared_results，因为 _shared_results 在同行
        # 并行执行时（如 FixAgent + ReviewAgent 同批次），可能尚未写入依赖任务的结果。
        prev = task_results.get(dep_id)
        if prev is None:
            prev = shared_results.get(dep_id)
        if prev is None:
            continue
        output = prev.get("output", {})
        content = (output.get("content", "") if isinstance(output, dict)
                   else str(output) if output else "")
        context[dep_id] = {
            "content": _truncate_text(content, budget_per_dep),
            "agent": prev.get("agent_type", ""),
            "type": output.get("type", "text") if isinstance(output, dict) else "text",
            "tokens_used": prev.get("tokens_used", 0),
            "metadata": output.get("metadata", {}) if isinstance(output, dict) else {},
        }
    return context if context else {}


# ──────────────────────────────────────────────
# 5. merge_and_finish 节点
# ──────────────────────────────────────────────


async def validate_plan_node(state: AgentGraphState) -> dict:
    """Layer 3 ValidationNode 入口。

    校验 plan 拓扑，标记 plan_valid 给 conditional_edge 路由。
    - valid=True → 继续 schedule_and_execute
    - valid=False → 触发 replan_plan_node

    注: parse_plan_with_reask 已完成"解析 + 拓扑校验 + LLM 自纠"，
    plan_tasks 里的 validate_plan 调用已移除。
    本节点做二次校验（防极端情况）+ frozen_plan 版本检查。
    """
    plan = state.get("plan")
    frozen_plan_data = state.get("frozen_plan")

    if not plan or not plan.get("tasks"):
        logger.warning("validate_plan_node: plan 为空, 标记无效")
        return {"plan_valid": False, "_validate_issues": ["plan 为空"]}

    issues: list[str] = []

    # 检查 frozen_plan 与当前 plan 是否一致 (防 plan 拓扑被中间环节意外修改)
    if frozen_plan_data:
        try:
            from .plan_frozen import FrozenPlan
            frozen = FrozenPlan.from_dict(frozen_plan_data)
            # 重新 hash 当前 plan, 与 frozen 的 version_hash 比较
            from .plan_validator import dict_to_plan
            current_plan = dict_to_plan(plan)
            current_frozen = FrozenPlan.freeze(current_plan)
            if current_frozen.version_hash != frozen.version_hash:
                issues.append(
                    f"plan 拓扑在 freeze 后被修改 "
                    f"(frozen={frozen.version_hash}, current={current_frozen.version_hash})"
                )
        except Exception as e:
            logger.debug("frozen_plan verify skipped", error=str(e)[:100])

    # 二次校验: 检查 agent_type 全部能在 registry 找到
    for t in plan.get("tasks", []):
        at = t.get("agent_type")
        if not at or get_card(at) is None:
            issues.append(f"task {t.get('id', '?')} 的 agent_type '{at}' 未注册")

    # 检查依赖图无环 (使用 FrozenPlan.freeze 的 DFS 环检测)
    task_ids = {t.get("id") for t in plan.get("tasks", [])}
    for t in plan.get("tasks", []):
        tid = t.get("id")
        deps = t.get("dependencies", []) or []
        if tid in deps:
            issues.append(f"task {tid} 存在自引用依赖")
        for d in deps:
            if d and d not in task_ids:
                issues.append(f"task {tid} 的 dep '{d}' 不在 plan 内")
    try:
        from src.agent.plan_frozen import FrozenPlan
        from .plan_validator import dict_to_plan
        current_pydantic_plan = dict_to_plan(plan)
        FrozenPlan.freeze(current_pydantic_plan)
    except ValueError as cycle_err:
        issues.append(str(cycle_err))
        logger.warning("validate_plan_node 检测到环形依赖: %s", cycle_err)

    valid = len(issues) == 0
    logger.info(
        "validate_plan_node: valid=%s issues=%d",
        valid, len(issues),
    )

    return {
        "plan_valid": valid,
        "_validate_issues": issues,
    }


async def replan_plan_node(state: AgentGraphState) -> dict:
    """replan 入口 - 当 validate_plan_node 失败时触发。

    保护已有 valid plan：
    - 如果 state["plan_valid"] 已经是 True，说明是 LangGraph 条件边路由误触发
      （validate_plan_node 返回 valid=True 但条件边读到了旧 state），
      此时不应重新规划，直接放行已有 plan。
    - 只有确实 plan_valid=False 时才执行真正的重规划。
    """
    from .plan_frozen import FrozenPlan

    # 保护已有 valid plan（防止 LangGraph 条件边路由误触发）
    if state.get("plan_valid") is True:
        logger.warning(
            "replan_plan_node 被误触发（plan_valid=True），跳过重规划，保留已有 plan"
        )
        return {
            "plan_valid": True,
            "_replan_protected": True,
        }

    frozen_data = state.get("frozen_plan")
    frozen = FrozenPlan.from_dict(frozen_data) if frozen_data else None

    if frozen and not frozen.can_upgrade(UpgradeLevel.REPLAN):
        logger.error(
            "replan 预算耗尽 (replans=%d/%d), 跳过重规划, 走现有 plan",
            frozen.replan_count, frozen.max_replans,
        )
        # FIX #7: 兜底放行, 但确保 plan 不为 None (走 fallback 兜底)
        if not state.get("plan") or not state.get("plan", {}).get("tasks"):
            logger.warning("replan 预算耗尽且无 plan, 注入 fallback plan")
            fallback = _build_fallback_plan(
                state.get("user_request", ""),
                state.get("understanding", {}).get("platform"),
            )
            return {
                "plan": fallback,
                "plan_valid": True,
                "_replan_skipped": True,
            }
        return {
            "plan_valid": True,
            "_replan_skipped": True,
        }

    if frozen:
        # FIX #6: 消费 REPLAN 预算, 保留原 frozen (而不是新 freeze)
        frozen.consume(UpgradeLevel.REPLAN)
        preserved_frozen_data = frozen.to_dict()
    else:
        preserved_frozen_data = None

    logger.info(
        "replan_plan_node: 触发重规划 (replans=%d/%d)",
        frozen.replan_count if frozen else 0,
        frozen.max_replans if frozen else 0,
    )

    # 重新调 plan_tasks — 清除旧 plan 防止被当作"预计算计划"跳过 LLM
    # 旧 plan 已有循环依赖，复用只会再次死锁
    state_copy = dict(state)
    state_copy.pop("plan", None)
    result = await plan_tasks(state_copy)

    # FIX #6: 保留原 frozen 计数 (而不是让 plan_tasks 覆盖)
    if preserved_frozen_data:
        result["frozen_plan"] = preserved_frozen_data
        result["plan_version"] = frozen.version_hash if frozen else None

    # FIX #7: replan 后 plan 仍为空 → 走 fallback
    if not result.get("plan") or not result.get("plan", {}).get("tasks"):
        logger.error("replan_plan_node: 重规划后 plan 仍为空, 注入 fallback plan")
        fallback = _build_fallback_plan(
            state.get("user_request", ""),
            state.get("understanding", {}).get("platform"),
        )
        result["plan"] = fallback
        result["plan_valid"] = True
        result["_replan_fallback"] = True

    result["plan_valid"] = True  # plan_tasks 已自校验过
    return result


async def merge_and_finish(state: AgentGraphState) -> dict:
    """合并所有子任务结果为最终输出 + 持久化"""
    tracer = _get_tracer()
    plan = state.get("plan") or {}
    # 合并 state 和共享结果（同一批次内任务的结果在 _shared_results 中）
    task_results = dict(state.get("task_results", {}))
    _shared = state.get("_shared_results", {})
    for _k, _v in _shared.items():
        if _k not in task_results:
            task_results[_k] = _v
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
            "xhs_post", "wechat", "wechat_article", "zhihu_article",
            "douyin", "bilibili", "weibo",
            "fiction", "storyboard", "script",
            "copywriting", "newsletter",
            "technical_doc", "tutorial",
        }

        primary_content = ""
        writing_draft = ""
        review_content = ""
        meta_sections = {}

        # 按优先级收集候选内容：FixAgent > ReviewFix > WritingAgent
        content_candidates = []
        # 修复：先构建完整结果池（同时查 task_results 和 _shared_results，避免 state 累积丢失）
        def _get_task_result(tid: str) -> dict:
            r = task_results.get(tid) or _shared.get(tid) or {}
            return r
        for task in task_queue:
            tid = task.get("id", "")
            result = _get_task_result(tid)
            output = result.get("output", {})
            agent_type = result.get("agent_type", task.get("agent_type", ""))
            task_topic = task.get("input", {}).get("topic", "")

            # FixAgent 的输出作为最高优先级最终内容
            # 但是：如果 FixAgent 输出的是 topic 标题而非正文（≤30字或与 topic 相同），
            # 说明 FixAgent 没有真正修改内容，应跳过，让 WritingAgent 初稿替补。
            if agent_type == "FixAgent" and isinstance(output, dict):
                content = output.get("content", "")
                if content:
                    is_topic_placeholder = (
                        len(content) <= 30
                        or (task_topic and content.strip() == task_topic.strip())
                        or len(content) < len(task_topic) * 0.5
                    )
                    if not is_topic_placeholder:
                        content_candidates.append(("fix", content))
                    else:
                        logger.warning("FixAgent 输出为 topic 占位符，跳过",
                                       topic=task_topic[:50], content_len=len(content))

            if agent_type == "ReviewAgent" and isinstance(output, dict):
                review_content = output.get("content", "")
                fixed = output.get("fixed_content", "")
                if output.get("rewrite_applied") and fixed:
                    content_candidates.append(("review_fix", fixed))


            if agent_type == "WritingAgent" and isinstance(output, dict):
                content = output.get("content", "")
                content_type = output.get("content_type", "")
                output_type = output.get("type", "")

                # ── 质量过滤 ──
                is_valid, reason = content_quality_check(content, min_chars=150)
                if not is_valid:
                    logger.warning(
                        "WritingAgent 输出质量不合格，跳过",
                        reason=reason,
                        content_len=len(content),
                        content_preview=content[:100],
                    )
                else:
                    writing_draft = content
                    if content_type in _CONTENT_TYPES or output_type == "text":
                        content_candidates.append(("writing", content))
                    else:
                        content_candidates.append(("writing_fallback", content))
            elif agent_type == "WritingAgent" and isinstance(output, str) and output:
                is_valid, reason = content_quality_check(output, min_chars=150)
                if not is_valid:
                    logger.warning("WritingAgent 字符串输出质量不合格", reason=reason)
                else:
                    writing_draft = output
                    content_candidates.append(("writing", output))

        # 按优先级选择主内容：fix > review_fix > writing
        # review-fix 循环时取最后一轮（倒序查找）
        for priority in ("fix", "review_fix", "writing", "writing_fallback"):
            for ptype, content in reversed(content_candidates):
                if ptype == priority:
                    primary_content = content
                    break
            if primary_content:
                break

        if not primary_content:
            if writing_draft:
                primary_content = writing_draft
            elif content_candidates:
                primary_content = content_candidates[-1][1]

        if not primary_content:
            logger.error("未找到主内容，合并失败")
            # FIX B: 这种情况也发 SSE error
            await _publish_streaming_event(state, "error", {
                "message": "Agent 执行完成但未生成有效内容",
            })
            return {
                "status": "failed",
                "error": "Agent 执行完成但未生成有效内容",
                "final_output": None,
            }
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

        # 提取标题：取正文第一行作为标题（去掉 Markdown 标记）
        _title = ""
        _first_line = primary_content.lstrip('\n').split('\n')[0].strip()
        _clean_line = re.sub(r'[#*`_\[\]()【】（）『』「」《》<>｜‖/\\]', '', _first_line).strip()
        if 2 <= len(_clean_line) <= 60:
            _title = _clean_line

        asyncio.create_task(_persist_results_async(
            state=state,
            sections=sections,
            full_text_parts=full_text_parts,
            status=status,
            title=_title,
            platform_code=(
                state.get("context", {}).get("platform")
                or plan.get("platform", "")
                or ""
            ),
        ))

        merge_span.set_attribute("merge.status", status)
        merge_span.set_status("OK")

        # FIX B: status=failed 时, 主动推 SSE error 事件 + 把 task 标 failed
        # 否则前端只能等 done, 且只看到空 result
        if status == "failed":
            failed_details = [
                f"{t.get('agent_type', '?')}({t.get('id', '?')[:8]})"
                for t in task_queue
                if t.get("id") in failed_ids
            ]
            error_msg = (
                f"所有任务都失败了 ({len(failed_ids)}/{len(task_queue)}): "
                f"{', '.join(failed_details[:5])}"
            )
            await _publish_streaming_event(state, "error", {"message": error_msg})
            # FIX B: 同时把 final_output 标 empty, 避免前端拿到半截 result
            empty_output = {
                "goal": plan.get("goal", ""),
                "plan_reasoning": plan.get("reasoning", ""),
                "platform": "",
                "title": "",
                "sections": [],
                "full_text": "",
                "review_content": "",
                "draft_content": None,
            }
            return {
                "final_output": empty_output,
                "status": status,
                "error": error_msg,
            }

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
                "platform": (
                    state.get("context", {}).get("platform")
                    or plan.get("platform", "")
                    or ""
                ),
                "title": _title,
                "sections": sections,
                "full_text": clean_text,
                "review_content": review_content,
                "draft_content": writing_draft or None,
            },
            "status": status,
        }


def _persist_results_sync(
    state: AgentGraphState,
    sections: list,
    full_text_parts: list,
    status: str,
    title: str = "",
    platform_code: str = "",
) -> None:
    """同步持久化执行结果到数据库"""
    try:
        from .persistence import save_agent_task, save_subtask, estimate_cost, save_content_from_agent_result

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
            conversation_id=state.get("context", {}).get("conversation_id", ""),
            prompt=state.get("user_request", ""),
            status=status,
            result={"sections": sections, "full_text": "".join(full_text_parts)},
            total_steps=len(state.get("task_queue", [])),
            completed_steps=len(state.get("completed_task_ids", [])),
            total_tokens=total_tokens,
            total_cost=estimate_cost(total_tokens, model_name),
            model_name=model_name,
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

        # ── 同步创建 Content 行（仅 completed / partial） ──
        # failed 任务没有可保存的正文，跳过；completed/partial 都值得进内容库供用户编辑
        if status in ("completed", "partial"):
            full_text = "".join(full_text_parts).strip()
            user_id = state.get("context", {}).get("user_id", "")
            if user_id and full_text:
                parsed_title = title
                parsed_tags: list[str] = []
                parsed_body = full_text

                # 优先尝试 JSON 解析（AI 按新 prompt 输出）
                json_text = full_text
                # 去掉 "思考：" 前缀（可选）
                think_match = re.match(r'^思考[：:]\s*', json_text)
                if think_match:
                    json_text = json_text[think_match.end():].strip()
                if json_text.startswith('{'):
                    try:
                        data = json.loads(json_text)
                        if isinstance(data, dict):
                            parsed_title = data.get("title", parsed_title)
                            raw_tags = data.get("tags", [])
                            if isinstance(raw_tags, list):
                                parsed_tags = [str(t) for t in raw_tags if t]
                            parsed_body = data.get("body", parsed_body)
                    except (json.JSONDecodeError, TypeError):
                        pass  # JSON 解析失败，走文本回退

                # 文本回退：从正文提取 #tag 标签并剥离
                if not parsed_tags:
                    tag_matches = re.findall(r'#[^\s#]+', parsed_body)
                    parsed_tags = list(dict.fromkeys(
                        t.lstrip('#') for t in tag_matches
                    ))
                    # 清理正文：去掉标题行 + 去掉 #tag 标签
                    title_skipped = False
                    clean_lines = []
                    for line in parsed_body.split('\n'):
                        stripped = line.strip()
                        if not title_skipped and stripped.startswith('# ') and len(stripped) > 2:
                            title_skipped = True
                            continue
                        if re.match(r'^#[^\s#]+(\s+#[^\s#]+)*$', stripped):
                            continue
                        clean_line = re.sub(r'#[^\s#]+', '', line).strip()
                        if clean_line:
                            clean_lines.append(clean_line)
                    parsed_body = '\n'.join(clean_lines).strip()

                save_content_from_agent_result(
                    user_id=user_id,
                    full_text=parsed_body,
                    title=parsed_title,
                    tags=parsed_tags,
                    topic=state.get("user_request", ""),
                    platform_code=platform_code,
                    source_task_id=task_id,
                    content_type=(
                        state.get("context", {}).get("content_type")
                        or state.get("context", {}).get("style", "")
                        or ""
                    ),
                    style=state.get("context", {}).get("style", ""),
                )
    except Exception as e:
        logger.warning("持久化失败: %s", e)


async def _persist_results_async(
    state: AgentGraphState,
    sections: list,
    full_text_parts: list,
    status: str,
    title: str = "",
    platform_code: str = "",
) -> None:
    """异步持久化执行结果到数据库"""
    try:
        await asyncio.to_thread(
            _persist_results_sync,
            state,
            sections,
            full_text_parts,
            status,
            title,
            platform_code,
        )
    except Exception as e:
        logger.warning("异步持久化失败: %s", e)


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
    completed_ids = set(state.get("completed_task_ids", []))
    failed_ids = set(state.get("failed_task_ids", []))
    total_tasks = len(task_queue)
    total_done = len(completed_ids) + len(failed_ids)

    if total_done >= total_tasks:
        return "done"

    # 当 ready 为空（execute 跑完一轮后被清空）时，**自己重算 ready**
    ready = state.get("ready_tasks", [])
    if not ready:
        # 用 schedule 同样的逻辑重算（避免 graph 自循环）
        ready = _compute_ready_tasks(task_queue, completed_ids, failed_ids)

    if ready:
        return "execute"

    # 重算仍空 → 真的是死锁
    logger.warning(
        "没有就绪任务但未全部完成，强制退出防止死锁",
        total_tasks=total_tasks, total_done=total_done,
    )
    return "done"
