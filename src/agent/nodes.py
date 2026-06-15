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
from .agent_utils import PLANNER_SYSTEM_PROMPT, extract_plan_from_llm_response, validate_plan_structure
from .platform_registry import PLATFORMS, detect_platform, get_names_map
from .handlers import HANDLER_MAP, handle_fallback
from .cache import agent_cache

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


def analyze_intent(user_request: str, context: dict | None = None) -> dict:
    """预处理用户输入：分析意图、检测平台、提取主题、推荐 Agent 组合

    在 ManagerAgent 规划之前运行，不依赖 LLM（纯规则+关键词）。
    输出结构化意图分析，供 LLM 规划 和 兜底计划 共同使用。

    Returns:
        dict: {
            "intent_type": str,        # create / research / analyze / edit / code / translate
            "platform": str | None,    # 检测到的平台代码
            "platform_name": str,      # 平台中文名
            "content_type": str,       # 内容类型
            "topic": str,              # 提取的主题
            "suggested_agents": list,  # 推荐的 Agent 组合
            "language": str,           # zh / en
            "constraints": {           # 约束条件
                "tone": str | None,
                "length": str | None,
                "style": str | None,
            }
        }
    """
    text = user_request.strip()
    lower = text.lower()

    # ── 1. 语言检测 ──
    has_chinese = bool(re.search(r'[一-鿿]', text))
    language = "zh" if has_chinese else "en"

    # ── 2. 平台检测（复用现有逻辑） ──
    platform = detect_platform(text)
    names_map = get_names_map()
    platform_name = names_map.get(platform, "") if platform else ""

    # ── 3. 内容类型推导 ──
    p = PLATFORMS.get(platform) if platform else None
    content_type = p.content_type if p else "blog"

    # ── 4. 意图分类 ──
    # 平台检测优先：检测到平台时 intent 偏向 create
    if platform:
        intent_type = "create"
    elif any(kw in lower for kw in ["搜索", "查一下", "search", "trending", "热点", "热搜", "最新"]):
        intent_type = "research"
    elif any(kw in lower for kw in ["分析", "总结", "评估", "analyze", "summarize", "review"]):
        intent_type = "analyze"
    elif any(kw in lower for kw in ["修改", "改进", "优化", "rewrite", "improve", "edit"]):
        intent_type = "edit"
    elif any(kw in lower for kw in ["代码", "脚本", "程序", "写一个", "code", "program", "function", "script"]):
        intent_type = "code"
        content_type = "code"
    elif any(kw in lower for kw in ["翻译", "translate", "译"]):
        intent_type = "translate"
        content_type = "text"
    else:
        intent_type = "create"

    # ── 5. 主题提取（取第一段有意义的话） ──
    # 去掉平台/意图关键词后的第一句作为主题
    topic = text
    lines = [l.strip() for l in text.replace('？', '?').replace('！', '!').split('\n') if l.strip()]
    for line in lines:
        # 找第一个长句作为主题
        for sep in ['?', '!', '。', '；', ';', '\n']:
            if sep in line:
                line = line.split(sep)[0]
        if len(line) >= 4:
            topic = line[:100]
            break

    # ── 6. Agent 组合推荐 ──
    if platform:
        from .platform_profile import get_suggested_agents
        suggested = list(get_suggested_agents(platform))
        # 保证写作和审核
        if "writing" not in suggested:
            suggested.insert(0, "writing")
        if "review" not in suggested:
            suggested.append("review")
        # 加 fix 做闭环
        if "fix" not in suggested and "review" in suggested:
            suggested.append("fix")
    elif intent_type == "research":
        suggested = ["search", "writing", "review", "fix"]
    elif intent_type == "analyze":
        suggested = ["search", "writing", "review"]
    elif intent_type == "edit":
        suggested = ["writing", "review", "fix"]
    elif intent_type == "code":
        suggested = ["code", "review"]
    elif intent_type == "translate":
        suggested = ["writing", "review"]
    else:
        suggested = ["writing", "review", "fix"]

    # ── 7. 约束提取（简单关键词匹配） ──
    constraints: dict = {"tone": None, "length": None, "style": None}
    tone_keywords = {
        "幽默": "humorous", "轻松": "humorous",
        "专业": "professional", "严肃": "professional",
        "友好": "friendly", "温暖": "warm", "亲切": "friendly",
        "深度": "deep", "技术": "technical",
    }
    for kw, val in tone_keywords.items():
        if kw in text:
            constraints["tone"] = val
            break
    if any(kw in lower for kw in ["短", "精简", "brief", "short"]):
        constraints["length"] = "short"
    elif any(kw in lower for kw in ["长", "详细", "深度", "long", "detailed"]):
        constraints["length"] = "long"
    else:
        constraints["length"] = "medium"

    return {
        "intent_type": intent_type,
        "platform": platform,
        "platform_name": platform_name,
        "content_type": content_type,
        "topic": topic,
        "suggested_agents": suggested,
        "language": language,
        "constraints": constraints,
    }


def _build_fallback_plan(user_request: str, platform_code: str | None = None,
                         intent_analysis: dict | None = None) -> dict:
    """动态兜底：优先用 intent_analysis，其次平台 profile，最后关键词"""
    goal = user_request[:200]

    p = None  # 平台 profile，可能在 if/else 中赋值，统一初始化防止 UnboundLocalError

    # 优先使用意图分析结果
    if intent_analysis:
        suggested = list(intent_analysis.get("suggested_agents", ["writing", "review"]))
        detected = intent_analysis.get("platform") or platform_code
        content_type = intent_analysis.get("content_type", "blog")
    else:
        detected = platform_code or detect_platform(user_request)
        p = PLATFORMS.get(detected) if detected else None
        content_type = p.content_type if p else "blog"

        # 从平台 profile 获取建议的 Agent 组合
        from .platform_profile import get_suggested_agents
        suggested = get_suggested_agents(detected) if detected else None

        if not suggested:
            text = user_request.lower()
            if any(kw in text for kw in ["代码", "脚本", "程序", "python", "java", "javascript", "function", "app"]):
                suggested = ["code", "review"]
            elif any(kw in text for kw in ["图表", "架构图", "流程图", "mermaid", "diagram"]):
                suggested = ["writing", "diagram", "review"]
            elif any(kw in text for kw in ["搜索", "热点", "热搜", "查", "最新", "趋势", "news", "trending"]):
                suggested = ["search", "writing", "review"]
            elif any(kw in text for kw in ["分析", "url", "链接", "网站", "网页", "抓取"]):
                suggested = ["search", "writing", "review"]
            else:
                suggested = ["writing", "review", "fix"]

    tasks = []
    last_id_index = -1

    # Agent type 映射表：short_name -> (agent_type, dependencies, priority)
    AGENT_MAP: dict[str, tuple[str, list[int], str, dict]] = {
        "writing": ("WritingAgent", [], "high", {"topic": user_request, "content_type": content_type, "platform": detected or ""}),
        "review": ("ReviewAgent", [], "medium", {"topic": user_request, "content_type": content_type, "platform": detected or ""}),
        "fix": ("FixAgent", [], "medium", {"topic": user_request, "content_type": content_type, "platform": detected or ""}),
        "seo": ("SEOAgent", [], "low", {"topic": user_request, "content_type": content_type, "platform": detected or "", "rewrite": True}),
        "search": ("WebSearch", [], "low", {"query": user_request, "count": 5}),
        "code": ("CodeAgent", [], "medium", {"task": user_request}),
        "diagram": ("DiagramAgent", [], "low", {"topic": user_request}),
    }

    built: list[dict] = []
    for short_name in suggested:
        if short_name not in AGENT_MAP:
            continue
        agent_type, deps_idx, priority, input_data = AGENT_MAP[short_name]

        # 解析依赖：列表中的数字表示前一个已构建任务的索引偏移
        resolved_deps = []
        for d in deps_idx:
            if d < len(built):
                resolved_deps.append(built[d]["id"])

        task = {
            "id": str(uuid.uuid4()), "agent_type": agent_type,
            "input": {**input_data, "platform": detected or ""},
            "dependencies": resolved_deps,
            "can_parallel_with": [], "retry_on_low_quality": agent_type in ("WritingAgent", "FixAgent"),
            "priority": priority,
        }
        built.append(task)

    # 自动构建依赖链：
    # - WritingAgent 依赖前一个 WebSearch
    # - review 依赖最后一个 writing 或 fix
    # - fix 依赖最后一个 review
    # - seo 依赖最后一个 writing 或 fix
    search_indices = [i for i, t in enumerate(built) if t["agent_type"] == "WebSearch"]
    writing_indices = [i for i, t in enumerate(built) if t["agent_type"] in ("WritingAgent", "FixAgent")]
    review_indices = [i for i, t in enumerate(built) if t["agent_type"] == "ReviewAgent"]
    fix_indices = [i for i, t in enumerate(built) if t["agent_type"] == "FixAgent"]

    for i, t in enumerate(built):
        if t["agent_type"] == "WritingAgent" and search_indices:
            target_id = built[search_indices[-1]]["id"]
            if target_id not in t["dependencies"]:
                t["dependencies"].append(target_id)
        if t["agent_type"] == "ReviewAgent" and writing_indices:
            target_id = built[writing_indices[-1]]["id"]
            if target_id not in t["dependencies"]:
                t["dependencies"].append(target_id)
        if t["agent_type"] == "FixAgent" and review_indices:
            target_id = built[review_indices[-1]]["id"]
            if target_id not in t["dependencies"]:
                t["dependencies"].append(target_id)
        if t["agent_type"] == "SEOAgent" and fix_indices:
            target_id = built[fix_indices[-1]]["id"]
            if target_id not in t["dependencies"]:
                t["dependencies"].append(target_id)
        elif t["agent_type"] == "SEOAgent" and writing_indices:
            target_id = built[writing_indices[-1]]["id"]
            if target_id not in t["dependencies"]:
                t["dependencies"].append(target_id)
        # WebSearch 无依赖，直接并行执行

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
    """复杂度分类 + 任务分解。LLM 动态规划，模板为兜底。"""
    tracer = _get_tracer()
    user_request = state.get("user_request", "")
    trace_id = str(uuid.uuid4())
    complexity = _classify_complexity(user_request)

    # 可用 Agent 清单（注入进 LLM 提示词，让 ManagerAgent 动态选择）
    AVAILABLE_AGENTS = """
可用 Agent 类型（根据任务需要选择合适的组合）：

1. WebSearch — 联网搜索热点、资料、数据。依赖：无。用于写作前收集素材。
2. URLCrawler — 抓取指定 URL 的内容。依赖：无。用于分析已有文章。
3. WritingAgent — 内容写作，根据主题和平台要求生成正文。依赖：无或搜索前置。
4. SEOAgent — SEO 优化，可改写正文使其更符合搜索习惯。依赖：正文前置。
5. CodeAgent — 代码生成，按需求编写代码。依赖：无。
6. DiagramAgent — 图表生成（Mermaid 格式）。依赖：无。
7. ReviewAgent — 内容审核，检查质量、违禁词、事实。依赖：正文前置。
8. FixAgent — 按审核意见自动修复内容。依赖：正文和审核前置。

常用组合示例：
- 写作任务：WebSearch → WritingAgent → ReviewAgent → FixAgent
- 纯搜索：WebSearch
- 代码任务：CodeAgent
- 改写优化：WritingAgent → SEOAgent
"""

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

        # ── 预处理：意图分析（纯规则，无 LLM） ──
        intent_analysis = analyze_intent(user_request, state.get("context"))
        detected = intent_analysis["platform"]
        names_map = get_names_map()
        platform = detected if detected else "generic"
        platform_name = intent_analysis["platform_name"] or names_map.get(platform, "通用内容")

        updates = {
            "trace_task_id": trace_id,
            "intent_analysis": intent_analysis,
            "start_time": time.time(),
            "thinking_steps": [],
            "status": "running",
        }

        span.add_event("step_started", {"step": "intent_analysis"})
        span.set_attribute("agent.intent_type", intent_analysis["intent_type"])
        span.set_attribute("agent.platform", platform)
        logger.info("意图分析：%s | 平台：%s | Agent：%s",
                     intent_analysis["intent_type"], platform_name, intent_analysis["suggested_agents"])

        updates.update(_add_thinking_step(
            state, "ManagerAgent", "意图分析完成",
            reasoning=f"识别到意图={intent_analysis['intent_type']}，平台={platform_name}，推荐Agent={intent_analysis['suggested_agents']}",
            input_summary=user_request[:200],
            output_summary=f"主题：{intent_analysis['topic'][:60]}",
        ))

        # ── 主路径：尝试 LLM 分解 ──
        llm = _get_llm(state, "ManagerAgent")
        if llm:
            try:
                context_str = json.dumps(state.get("context", {}), ensure_ascii=False, indent=2) if state.get("context") else "{}"
                # 结构化意图分析注入 LLM 提示词
                analysis_str = json.dumps(intent_analysis, ensure_ascii=False, indent=2)
                # 15 秒超时
                response = await asyncio.wait_for(
                    llm.ainvoke([
                        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                        HumanMessage(content=f"""## 用户原始请求
{user_request}

## 意图分析（预处理结果）
{analysis_str}

## Agent 清单
{AVAILABLE_AGENTS}

复杂度：{complexity}
上下文：{context_str}

请根据用户请求和意图分析，分解为子任务列表（JSON格式），选择合适的 Agent 组合。
注意：intent_analysis.suggested_agents 是推荐组合，你可以根据需要调整。"""),
                    ]),
                    timeout=60.0,
                )
                plan = _parse_plan(response.content, user_request, platform=detected)

                if len(plan.get("tasks", [])) == 0:
                    logger.warning("LLM 计划没有任务，使用兜底方案")
                    plan = _build_fallback_plan(user_request, intent_analysis=intent_analysis)
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
                logger.warning("LLM 分解失败，使用模板兜底", error=str(e)[:200], exc_type=type(e).__name__)
                span.record_exception(e)

        # 兜底路径：LLM 不可用或分解失败时使用模板
        plan = _build_fallback_plan(user_request, intent_analysis=intent_analysis)
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

            # Emit agent_start 到控制台
            emitter = get_agent_event_emitter()
            await emitter.emit(AgentEvent(
                event_type="agent_start",
                agent_name=agent_type,
                message=str(task_input.get("topic", task_input.get("query", task_input.get("task", ""))))[:100] or "开始执行",
                run_id=task_id or agent_type,
                llm_model=getattr(llm, 'model_name', '') if llm else '',
            ))

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

        # 按优先级选择主内容：fix > review_fix > seo_rewrite > writing
        for priority in ("fix", "review_fix", "seo_rewrite", "writing"):
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
                    # 检查是否有修正版内容
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
