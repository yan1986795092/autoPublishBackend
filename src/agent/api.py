from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError
from typing import Optional, AsyncGenerator, Any
import asyncio
import json
import structlog
import time
import uuid

logger = structlog.get_logger(__name__)

# ===================== SSE 流式接口 =====================


from sse_starlette.sse import EventSourceResponse

from .schemas import AgentTask
from .factory import create_orchestrator
from .orchestrator import Orchestrator
from .workflow.task_queue import task_queue, event_bus, StreamEvent
from .workflow.worker import worker
from .cache import agent_cache, llm_cache
from src.api.auth import get_current_user
from src.core.security import decode_token
from src.models import User
from src.tools.vector_search import vector_search
from src.config import get_settings
from src.core.request_guards import ensure_task_owner, validate_public_http_url

router = APIRouter(tags=["agent"])


# ─────────────────────────────────────────────────────────
# Result Schema 收敛（spec 2026-07-01）
# ─────────────────────────────────────────────────────────

def _safe_output_get(output: Any, key: str, default: Any = None) -> Any:
    """从 result.output 取字段，兼容 output 为 None / 非 dict 的情况"""
    if not isinstance(output, dict):
        return default
    return output.get(key, default)


def _safe_usage_get(usage: Any, key: str, default: Any = None) -> Any:
    """从 result.usage 取字段"""
    if not isinstance(usage, dict):
        return default
    return usage.get(key, default)


def _transform_result_for_frontend(result: dict | None) -> dict:
    """把 AgentResult 收敛成"实时 SSE 业务轨"schema

    设计要点（spec 2026-07-01 §3.3）：
    - 保留 result.output.* 业务字段
    - 保留 result.usage 部分字段（MessageBubble 实际渲染）
    - **不返回** 运行时缓冲字段（_stages / _thinkingSteps / _llmChunks / _reasoningChunks / _draftContent）
    - **不返回** agent_events 全量 dump
    """
    result = result or {}
    output = result.get("output") if isinstance(result, dict) else None
    usage = result.get("usage") if isinstance(result, dict) else None

    return {
        # 标识
        "task_id": result.get("task_id") if isinstance(result, dict) else None,
        "status": result.get("status") if isinstance(result, dict) else None,
        "agent_type": result.get("agent_type") if isinstance(result, dict) else None,
        # 产物
        "output": {
            "full_text": _safe_output_get(output, "full_text") or _safe_output_get(output, "content"),
            "title": _safe_output_get(output, "title"),
            "platform": _safe_output_get(output, "platform"),
            "content_type": _safe_output_get(output, "content_type"),
            "draft_content": _safe_output_get(output, "draft_content"),
            "review_content": _safe_output_get(output, "review_content"),
        },
        # UI 摘要（小卡片显示）
        "usage": {
            "tokens_used": _safe_usage_get(usage, "tokens_used"),
            "time_elapsed": _safe_usage_get(usage, "time_elapsed"),
            "model": _safe_usage_get(usage, "model"),
        },
        # 故意不带：
        # - _stages / _thinkingSteps / _llmChunks / _reasoningChunks（运行时缓冲，前端 store 自己累积）
        # - agent_events（全量 dump，仅 DB 持久化用）
        # - error（顶层状态由 status 表达）
    }


def _result_for_persistence(
    frontend_result: dict,
    *,
    stages: list | None = None,
    thinking_steps: list | None = None,
    llm_chunks: dict | None = None,
    reasoning_chunks: dict | None = None,
    draft_content: str | None = None,
    agent_events: dict | None = None,
) -> dict:
    """把"运行时缓冲"拼到收敛后的 result，得到 DB 持久化 schema

    关键约束：
    - **输入** frontend_result 必须来自 _transform_result_for_frontend()
    - **输出** 入库 schema 与历史消息兼容（前端读出时已有 fallback 逻辑）
    """
    persisted = dict(frontend_result)  # 浅拷贝（output / usage 是 dict，再深拷一层）
    persisted["output"] = dict(frontend_result.get("output") or {})
    persisted["usage"] = dict(frontend_result.get("usage") or {})

    if stages is not None:
        persisted["_stages"] = stages
    if thinking_steps is not None:
        persisted["_thinkingSteps"] = thinking_steps
    if llm_chunks is not None:
        persisted["_llmChunks"] = llm_chunks
    if reasoning_chunks is not None:
        persisted["_reasoningChunks"] = reasoning_chunks
    if draft_content is not None:
        persisted["_draftContent"] = draft_content
    if agent_events is not None:
        persisted["agent_events"] = agent_events

    return persisted


class AgentRunRequest(BaseModel):
    prompt: str
    context: Optional[dict] = None
    skill_id: Optional[str] = None  # 应用指定的 Skill ID
    platform: Optional[str] = None  # 目标平台代码，如 xiaohongshu
    trigger: Optional[dict] = None  # 对话引擎的 trigger dict（透传给 skill_context）


class AgentRunResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


class AgentAsyncSubmitRequest(BaseModel):
    prompt: str
    skill_id: Optional[str] = None
    platform: Optional[str] = None  # 目标平台代码
    trigger: Optional[dict] = None  # 对话引擎的 trigger dict


class AgentAsyncStatusResponse(BaseModel):
    success: bool
    data: Optional[dict] = None


def _create_orchestrator(**kwargs) -> Orchestrator:
    """创建 Orchestrator 实例（通过 ModelRouter 为各 Agent 分配独立 LLM 配置）"""
    return create_orchestrator(**kwargs)


# ===================== 同步接口 =====================


@router.post("/agent/run", response_model=AgentRunResponse)
async def run_agent(
    request: AgentRunRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user),
):
    """运行 Agent 编排：输入需求，返回 Multi-Agent 协作结果"""
    try:
        context = request.context or {}
        context["user_id"] = current_user.id
        # 注入 HTTP trace_id 用于端到端关联
        context["trace_id"] = http_request.headers.get("X-Trace-Id", "")

        # Skill 注入（与 /stream 端点保持一致）
        if request.skill_id:
            from src.api.skills import load_skill_full_for_user
            from src.database import SessionLocal
            db = SessionLocal()
            try:
                skill_data = load_skill_full_for_user(request.skill_id, db, current_user.id)
                if skill_data:
                    context.update({
                        "skill_id": request.skill_id,
                        "skill_name": skill_data.get("name", ""),
                        "skill_system_prompt": skill_data.get("system_prompt", ""),
                        "skill_preset_plan": skill_data.get("preset_plan"),
                        "skill_user_template": skill_data.get("user_prompt_template", ""),
                    })
                    # Skill template 转换
                    if skill_data.get("user_prompt_template"):
                        template = skill_data["user_prompt_template"]
                        template = template.replace("{topic}", request.prompt.split("\n")[0])
                        template = template.replace("{content}", request.prompt)
                        prompt_to_use = template
                    else:
                        prompt_to_use = request.prompt
                else:
                    prompt_to_use = request.prompt
            finally:
                db.close()
        else:
            prompt_to_use = request.prompt

        # 平台注入
        if request.platform:
            context["platform"] = request.platform
            from .platform_profile import get_profile
            profile = get_profile(request.platform)
            if profile:
                context["platform_name"] = profile.get("name", request.platform)
                context["platform_system_prompt"] = profile.get("system_prompt", "")
                context["platform_writing_rules"] = profile.get("writing_rules", {})
                context["platform_quality_check"] = profile.get("quality_check", "")

        orchestrator = _create_orchestrator()
        result = await orchestrator.run(
            user_request=prompt_to_use,
            context=context,
            timeout=240.0,  # 4 分钟全局超时
        )
        return AgentRunResponse(
            success=result.status.value in ("completed", "partial"),
            data=result.to_dict(),
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Agent 任务超时，请尝试简化需求或使用异步接口")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 错误: {str(e)}")


@router.post("/agent/plan", response_model=AgentRunResponse)
async def preview_plan(
    request: AgentRunRequest,
    current_user: User = Depends(get_current_user),
):
    """预览 Agent 执行计划（不实际执行，只展示拆解结果）"""
    try:
        context = request.context or {}
        context["user_id"] = current_user.id

        # Skill 注入
        if request.skill_id:
            from src.api.skills import load_skill_full_for_user
            from src.database import SessionLocal
            db = SessionLocal()
            try:
                skill_data = load_skill_full_for_user(request.skill_id, db, current_user.id)
                if skill_data:
                    context.update({
                        "skill_id": request.skill_id,
                        "skill_name": skill_data.get("name", ""),
                        "skill_system_prompt": skill_data.get("system_prompt", ""),
                        "skill_preset_plan": skill_data.get("preset_plan"),
                        "skill_user_template": skill_data.get("user_prompt_template", ""),
                    })
                    if skill_data.get("user_prompt_template"):
                        template = skill_data["user_prompt_template"]
                        template = template.replace("{topic}", request.prompt.split("\n")[0])
                        template = template.replace("{content}", request.prompt)
                        prompt_to_use = template
                    else:
                        prompt_to_use = request.prompt
                else:
                    prompt_to_use = request.prompt
            finally:
                db.close()
        else:
            prompt_to_use = request.prompt

        # 平台注入
        if request.platform:
            context["platform"] = request.platform
            from .platform_profile import get_profile
            profile = get_profile(request.platform)
            if profile:
                context["platform_name"] = profile.get("name", request.platform)
                context["platform_system_prompt"] = profile.get("system_prompt", "")
                context["platform_writing_rules"] = profile.get("writing_rules", {})

        orchestrator = _create_orchestrator()
        plan = await orchestrator._decompose(
            user_request=prompt_to_use,
            context=context,
        )
        return AgentRunResponse(
            success=True,
            data={
                "goal": plan.goal,
                "reasoning": plan.reasoning,
                "estimated_steps": plan.estimated_steps,
                "tasks": [
                    {
                        "agent_type": t.agent_type,
                        "input": t.input,
                        "dependencies": t.dependencies,
                    }
                    for t in plan.tasks
                ],
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===================== 内容再利用 Pipeline =====================


class RepurposeRequest(BaseModel):
    content: str
    title: str = ""
    formats: list[str] = ["social_post", "newsletter", "diagram"]
    style: str = ""
    platforms: list[str] = []  # 可选：同时适配到指定平台


REPURPOSE_AGENT_MAP = {
    "social_post": "WritingAgent",
    "newsletter": "WritingAgent",
    "diagram": "DiagramAgent",
    "copywriting": "WritingAgent",
    "script": "WritingAgent",
    "report": "WritingAgent",
    "weibo_post": "WritingAgent",
    "twitter_thread": "WritingAgent",
    "podcast_script": "WritingAgent",
}

REPURPOSE_FORMAT_GUIDE = {
    "social_post": "将其转化为 3-5 条适合社交媒体的短帖，每条100-200字，带互动引导和emoji",
    "newsletter": "将其改写为一封邮件Newsletter，包含友好的问候、摘要和行动号召",
    "diagram": "从内容中提取关键概念，生成一份Mermaid.js知识架构图",
    "copywriting": "将其改写为营销文案，突出价值主张和行动号召",
    "script": "将其改写为短视频口播脚本，包含开场钩子和分镜描述",
    "report": "将其扩展为一份结构化的分析报告，包含数据摘要和结论建议",
    "weibo_post": "将其提炼为 2-3 条微博短帖，每条不超过140字，带话题标签，一针见血",
    "twitter_thread": "将其改写为英文 Twitter/X Thread，3-6条推文，开场hook吸引点击，每条独立可读",
    "podcast_script": "将其转化为一期播客脚本，包含开场白、2-3个讨论环节、嘉宾问答、收尾总结",
}


@router.post("/agent/repurpose", response_model=AgentRunResponse)
async def repurpose_content(
    request: RepurposeRequest,
    current_user: User = Depends(get_current_user),
):
    """内容再利用 Pipeline（§5.3）：一篇内容 → 多种形态输出

    将一篇长文自动转化为：社交媒体帖、Newsletter、架构图、文案等多种形态。
    支持同时适配到不同平台格式。
    """
    try:
        orchestrator = _create_orchestrator()
        sections = []
        source_content = request.content
        source_title = request.title or source_content[:50]

        for fmt in request.formats:
            agent_type = REPURPOSE_AGENT_MAP.get(fmt, "WritingAgent")
            format_guide = REPURPOSE_FORMAT_GUIDE.get(fmt, "")

            task_input = {
                "topic": source_title,
                "content": source_content,
                "content_type": fmt,
                "style": request.style or "auto",
            }

            # 构建 repurpose prompt
            prompt = f"""## 内容再利用任务

将以下内容转化为 {fmt} 格式。

转化指引：{format_guide}

风格：{request.style or '保持原文风格'}

原始标题：{source_title}

原始内容：
{source_content[:5000]}

请先写 REASONING，然后输出 OUTPUT。"""

            subtask = await orchestrator._fallback_llm_call(
                task=AgentTask(
                    id=str(uuid.uuid4()),
                    agent_type=agent_type,
                    input={**task_input, "task": prompt},
                    context={"original_content": source_content},
                ),
                context={},
                previous_results={},
            )

            output_content = subtask.output.get("content", "")

            # 如果指定了平台，进一步做平台适配
            platform_outputs = {}
            if request.platforms and output_content:
                for platform_code in request.platforms:
                    try:
                        platform_task = await orchestrator._fallback_llm_call(
                            task=AgentTask(
                                id=str(uuid.uuid4()),
                                agent_type="PlatformAdapter",
                                input={
                                    "topic": source_title,
                                    "content": output_content,
                                    "platform": platform_code,
                                    "task": f"将以下内容适配为 {platform_code} 平台格式",
                                },
                            ),
                            context={},
                            previous_results={},
                        )
                        platform_outputs[platform_code] = platform_task.output.get("content", output_content)
                    except Exception as e:
                        logger.warning("平台适配失败", _type="system", event_name="platform_adapt_fail",
                                      message=f"平台适配失败 {platform_code}: {e}", status="error")
                        platform_outputs[platform_code] = output_content

            sections.append({
                "format": fmt,
                "agent": agent_type,
                "content": output_content,
                "status": subtask.status.value,
                "platform_versions": platform_outputs if platform_outputs else None,
            })

        return AgentRunResponse(
            success=True,
            data={
                "source_title": source_title,
                "source_length": len(source_content),
                "outputs": sections,
                "formats_count": len(sections),
                "platforms": request.platforms,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===================== 异步接口 =====================


@router.post("/agent/async/submit")
async def submit_agent_task(
    request: AgentAsyncSubmitRequest,
    current_user: User = Depends(get_current_user),
):
    """提交异步 Agent 任务（适合长时间执行）"""
    skill_context: dict = {}
    if request.skill_id:
        from src.api.skills import load_skill_full_for_user
        from src.database import SessionLocal
        db = SessionLocal()
        try:
            skill_data = load_skill_full_for_user(request.skill_id, db, current_user.id)
            if skill_data:
                skill_context = {
                    "skill_id": request.skill_id,
                    "skill_name": skill_data.get("name", ""),
                    "skill_system_prompt": skill_data.get("system_prompt", ""),
                    "skill_preset_plan": skill_data.get("preset_plan"),
                    "skill_user_template": skill_data.get("user_prompt_template", ""),
                }
                if skill_data.get("user_prompt_template"):
                    template = skill_data["user_prompt_template"]
                    template = template.replace("{topic}", request.prompt.split("\n")[0])
                    template = template.replace("{content}", request.prompt)
                    prompt_to_use = template
                else:
                    prompt_to_use = request.prompt
            else:
                prompt_to_use = request.prompt
        finally:
            db.close()
    else:
        prompt_to_use = request.prompt

    async_task = await task_queue.enqueue(
        prompt=prompt_to_use,
        user_id=current_user.id,
        skill_context=skill_context if skill_context else None,
    )
    return {"success": True, "data": {"task_id": async_task.id}}


@router.get("/agent/async/status/{task_id}", response_model=AgentAsyncStatusResponse)
async def get_agent_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user),
):
    """查询异步任务状态"""
    task = ensure_task_owner(task_queue.get_task(task_id), current_user)
    return AgentAsyncStatusResponse(
        success=True,
        data=task.to_dict(),
    )


@router.get("/agent/async/list")
async def list_agent_tasks(
    current_user: User = Depends(get_current_user),
    limit: int = 20,
):
    """查询当前用户的任务列表"""
    tasks = task_queue.list_tasks(user_id=current_user.id, limit=limit)
    return {
        "success": True,
        "data": {
            "tasks": [t.to_dict() for t in tasks],
            "total": len(tasks),
        },
    }


@router.delete("/agent/async/{task_id}")
async def cancel_agent_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
):
    """取消异步任务"""
    ensure_task_owner(task_queue.get_task(task_id), current_user)
    success = task_queue.cancel_task(task_id)
    return {
        "success": success,
        "data": {"task_id": task_id, "state": "cancelled"},
    }


# ===================== SSE 流式接口 =====================


class AgentStreamRequest(BaseModel):
    prompt: str
    context: Optional[dict] = None
    skill_id: Optional[str] = None  # 应用指定的 Skill ID
    platform: Optional[str] = None  # 目标平台代码，如 xiaohongshu
    parameters: Optional[dict] = None  # 前端参数（tone、creativity、audience 等）
    trigger: Optional[dict] = None  # 对话引擎的 trigger dict（透传给 skill_context）


@router.post("/agent/stream", deprecated=True)
async def agent_stream(
    request: AgentStreamRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user),
):
    """⚠️ DEPRECATED — 2026-09-30 下线，请改用 /api/v1/agent/stream/v2

    v1 流式 Agent 执行 — 通过 EventBus 实时推送计划、思考步骤、进度和结果。

    与 v2 的差异：
    - v1 下发 `thinking_step / reasoning` 事件（v2 过滤掉）
    - v1 result 数据是 AgentResult 全量 dump（v2 用收敛 schema）

    支持事件类型：plan, task_created, thinking_step, llm_chunk, reasoning, progress, result, error, done
    """
    # 提取 HTTP trace_id 用于端到端关联
    http_trace_id = http_request.headers.get("X-Trace-Id", "")
    task_id_str = ""  # 实际 task_id 在入队后设置

    async def event_generator() -> AsyncGenerator[dict, None]:
        nonlocal task_id_str
        try:
            # 0. 加载 Skill（如果指定了 skill_id）
            skill_context: dict = {}
            if request.skill_id:
                from src.api.skills import load_skill_full_for_user
                from src.database import SessionLocal
                db = SessionLocal()
                try:
                    skill_data = load_skill_full_for_user(request.skill_id, db, current_user.id)
                    if skill_data:
                        skill_context = {
                            "skill_id": request.skill_id,
                            "skill_name": skill_data.get("name", ""),
                            "skill_system_prompt": skill_data.get("system_prompt", ""),
                            "skill_preset_plan": skill_data.get("preset_plan"),
                            "skill_preset_complexity": skill_data.get("preset_plan", {}).get("complexity", "medium"),
                            "skill_user_template": skill_data.get("user_prompt_template", ""),
                        }
                    else:
                        logger.warning("未找到 Skill", _type="system", event_name="skill_not_found",
                                      message=f"找不到 Skill: {request.skill_id}", status="error")
                finally:
                    db.close()

            # 应用 Skill 的 user_prompt_template 转换
            user_prompt, platform_code = request.prompt, request.platform
            if skill_context.get("skill_user_template"):
                template = skill_context["skill_user_template"]
                template = template.replace("{topic}", user_prompt.split("\n")[0])
                template = template.replace("{content}", user_prompt)
                user_prompt = template

            # 前端参数注入
            if request.parameters:
                for k, v in request.parameters.items():
                    if v is not None:
                        skill_context[k] = v

            # ── 对话 trigger 注入（来自 /api/v1/conversation/chat）──
            # 把 trigger dict 合并到 skill_context，让 Agent 任务拼装时拿到 platform/style/length 等
            if request.trigger and isinstance(request.trigger, dict):
                for k in ("platform", "style", "length", "tone", "audience",
                          "keywords", "revision_focus", "action", "topic"):
                    v = request.trigger.get(k)
                    if v not in (None, "", [], {}):
                        skill_context[k] = v
                # user_prompt 优先用 trigger.topic（清洗后），没有则用 request.prompt
                trigger_topic = (request.trigger.get("topic") or "").strip()
                if trigger_topic and 3 <= len(trigger_topic) <= 500:
                    user_prompt = trigger_topic
                elif request.trigger.get("_prompt"):
                    user_prompt = request.trigger["_prompt"]
                # 兜底标记透传，WritingAgent 节点可识别
                if request.trigger.get("_fallback"):
                    skill_context["_trigger_fallback"] = True

            # 平台检测（轻量：字符串匹配，无需 Orchestrator / LLM）
            if not platform_code:
                from .platform_registry import detect_platform
                platform_code = detect_platform(user_prompt)
            if platform_code:
                skill_context["platform"] = platform_code
                # 从 platform_profile 加载平台名称
                from .platform_profile import get_profile
                profile = get_profile(platform_code)
                if profile:
                    skill_context["platform_name"] = profile.get("name", platform_code)
                    skill_context["platform_system_prompt"] = profile.get("system_prompt", "")
                    skill_context["platform_writing_rules"] = profile.get("writing_rules", {})
                    skill_context["platform_agent_pipeline"] = profile.get("agent_pipeline", {})
                    skill_context["platform_quality_check"] = profile.get("quality_check", "")

            # 不再预计算计划——让 ManagerAgent LLM 在 Graph 中动态规划
            # 但仍然 yield 一个轻量占位事件，让前端显示"规划中"
            yield {
                "event": "plan",
                "data": json.dumps({
                    "goal": user_prompt[:200],
                    "reasoning": "ManagerAgent 正在动态规划最佳 Agent 组合...",
                    "estimated_steps": 0,
                    "tasks": [],
                }, ensure_ascii=False),
            }

            # 2. 创建 EventBus 订阅队列（线程安全队列用于跨协程传递事件）
            evt_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

            async def on_event(event: StreamEvent) -> None:
                """EventBus 回调：将事件放入队列"""
                await evt_queue.put(event)

            # 3. 先预留 task_id 并订阅，再入队，避免快速任务丢失首批事件
            plan = None
            # 注入 HTTP trace_id 到 skill_context 用于端到端关联
            if http_trace_id:
                skill_context["trace_id"] = http_trace_id
            task_id_str = task_queue.reserve_task_id()
            event_bus.subscribe(task_id_str, on_event)

            async_task = await task_queue.enqueue(
                prompt=user_prompt,
                user_id=current_user.id,
                plan=plan,
                skill_context=skill_context if skill_context else None,
                task_id=task_id_str,
            )
            task_id_str = async_task.id

            yield {
                "event": "task_created",
                "data": json.dumps({"task_id": task_id_str}, ensure_ascii=False),
            }

            # 5. 收集已完成推送的步骤索引（防止重复）
            seen_step_count = 0

            try:
                timeout_seconds = 600
                start_time = asyncio.get_event_loop().time()

                while True:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > timeout_seconds:
                        yield {
                            "event": "error",
                            "data": json.dumps({"message": "任务执行超时"}, ensure_ascii=False),
                        }
                        break

                    # 等待事件（带超时，周期性检查任务是否完成）
                    try:
                        event = await asyncio.wait_for(evt_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        task = task_queue.get_task(task_id_str)
                        if not task:
                            break
                        if task.state.value not in ("completed", "failed", "cancelled"):
                            continue

                        # ── 任务已结束：先排空事件队列（避免 _publish_event 延迟导致丢失事件）──
                        remaining_events: list[StreamEvent] = []
                        while True:
                            try:
                                remaining_events.append(evt_queue.get_nowait())
                            except asyncio.QueueEmpty:
                                break

                        got_result = False
                        for revt in remaining_events:
                            if revt.event_type == "result":
                                yield {"event": "result", "data": json.dumps(revt.data, ensure_ascii=False)}
                                got_result = True
                            elif revt.event_type == "error":
                                yield {"event": "error", "data": json.dumps(revt.data, ensure_ascii=False)}
                                got_result = True

                        # ── 事件队列中没有 result/error，直接从 task 提取 ──
                        if not got_result:
                            if task.state.value == "completed" and task.result:
                                yield {
                                    "event": "result",
                                    "data": json.dumps(task.result, ensure_ascii=False),
                                }
                            elif task.state.value == "failed":
                                yield {
                                    "event": "error",
                                    "data": json.dumps({"message": task.error or "任务执行失败"}, ensure_ascii=False),
                                }

                        yield {
                            "event": "done",
                            "data": json.dumps({"task_id": task_id_str, "state": task.state.value}, ensure_ascii=False),
                        }
                        break

                    et = event.event_type

                    # 跳过已处理的 steps（由 task.result 包含的完整 thinking_trace）
                    if et == "thinking_step":
                        seen_step_count += 1
                        yield {
                            "event": "thinking_step",
                            "data": json.dumps(event.data, ensure_ascii=False),
                        }
                    elif et == "progress":
                        yield {
                            "event": "progress",
                            "data": json.dumps(event.data, ensure_ascii=False),
                        }
                    elif et == "stage":
                        yield {
                            "event": "stage",
                            "data": json.dumps(event.data, ensure_ascii=False),
                        }
                    elif et == "llm_chunk":
                        yield {
                            "event": "llm_chunk",
                            "data": json.dumps(event.data, ensure_ascii=False),
                        }
                    elif et == "reasoning":
                        yield {
                            "event": "reasoning",
                            "data": json.dumps(event.data, ensure_ascii=False),
                        }
                    elif et == "result":
                        yield {
                            "event": "result",
                            "data": json.dumps(event.data, ensure_ascii=False),
                        }
                        yield {
                            "event": "done",
                            "data": json.dumps({"task_id": task_id_str, "state": "completed"}, ensure_ascii=False),
                        }
                        break
                    elif et == "error":
                        yield {
                            "event": "error",
                            "data": json.dumps(event.data, ensure_ascii=False),
                        }
                        yield {
                            "event": "done",
                            "data": json.dumps({"task_id": task_id_str, "state": "failed"}, ensure_ascii=False),
                        }
                        break

            finally:
                # 取消订阅 EventBus（task_id_str 已在 enqueue 后赋值）
                if task_id_str:
                    event_bus.unsubscribe(task_id_str, on_event)

        except Exception as e:
            logger.exception("SSE 流错误", _type="system", event_name="sse_error",
                             message=f"SSE 流错误: {e}", status="error")
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


# ===================== 业务 SSE v2（spec 2026-07-01） =====================
# 与 /agent/stream 的核心差异：
# 1. 实时 SSE 不下发 thinking_step / reasoning（前端 store 运行时自累积，无需服务端重复推送）
# 2. result 事件用 _transform_result_for_frontend() 收敛（不携带运行时缓冲字段）
# 3. 其余业务事件（plan / stage / progress / llm_chunk / done / error）保持行为一致


@router.post("/agent/stream/v2")
async def agent_stream_v2(
    request: AgentStreamRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user),
):
    """业务轨 SSE v2 — 实时流只下发"前端真正需要"的事件类型与字段

    与 v1 的差异（spec 2026-07-01 §3）：
    - **不下发** `thinking_step / reasoning`（前端 store 自累积）
    - `result` 事件数据用收敛 schema，**不携带** `_stages / _thinkingSteps / _llmChunks / _reasoningChunks / _draftContent / agent_events`
    - 行为保持：plan / stage / progress / llm_chunk / done / error 与 v1 一致

    迁移路径：v2 上线后，v1 标 deprecated；2026-09-30 删除 v1。
    """
    http_trace_id = http_request.headers.get("X-Trace-Id", "")
    task_id_str = ""

    async def event_generator() -> AsyncGenerator[dict, None]:
        nonlocal task_id_str

        # 复用 v1 的 skill/prompt/trigger 初始化
        skill_context, user_prompt = _init_stream_context(
            request=request,
            current_user=current_user,
            http_trace_id=http_trace_id,
        )

        try:
            # 复用 v1 的 plan 占位事件
            yield {
                "event": "plan",
                "data": json.dumps({
                    "goal": user_prompt[:200],
                    "reasoning": "ManagerAgent 正在动态规划最佳 Agent 组合...",
                    "estimated_steps": 0,
                    "tasks": [],
                }, ensure_ascii=False),
            }

            evt_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

            async def on_event(event: StreamEvent) -> None:
                await evt_queue.put(event)

            task_id_str = task_queue.reserve_task_id()
            event_bus.subscribe(task_id_str, on_event)

            await task_queue.enqueue(
                prompt=user_prompt,
                user_id=current_user.id,
                plan=None,
                skill_context=skill_context if skill_context else None,
                task_id=task_id_str,
            )

            yield {
                "event": "task_created",
                "data": json.dumps({"task_id": task_id_str}, ensure_ascii=False),
            }

            # 业务轨过滤：忽略 thinking_step / reasoning（前端 store 自累积）
            # 其余业务事件实时下发。result 用收敛 schema。
            seen_step_count = 0
            timeout_seconds = 600
            start_time = asyncio.get_event_loop().time()

            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout_seconds:
                    yield {
                        "event": "error",
                        "data": json.dumps({"message": "任务执行超时"}, ensure_ascii=False),
                    }
                    break

                try:
                    event = await asyncio.wait_for(evt_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    task = task_queue.get_task(task_id_str)
                    if not task:
                        break
                    if task.state.value not in ("completed", "failed", "cancelled"):
                        continue

                    # 兜底分支：等待超时且任务已结束
                    remaining = []
                    while True:
                        try:
                            remaining.append(evt_queue.get_nowait())
                        except asyncio.QueueEmpty:
                            break

                    got_result = False
                    for revt in remaining:
                        # 业务轨过滤：跳过 internal-only events
                        if _should_skip_in_business_track(revt):
                            continue
                        if revt.event_type == "result":
                            yield {
                                "event": "result",
                                "data": json.dumps(
                                    _transform_result_for_frontend(revt.data),
                                    ensure_ascii=False,
                                ),
                            }
                            got_result = True
                        elif revt.event_type == "error":
                            yield {
                                "event": "error",
                                "data": json.dumps(revt.data, ensure_ascii=False),
                            }
                            got_result = True

                    if not got_result:
                        if task.state.value == "completed" and task.result:
                            yield {
                                "event": "result",
                                "data": json.dumps(
                                    _transform_result_for_frontend(task.result),
                                    ensure_ascii=False,
                                ),
                            }
                        elif task.state.value == "failed":
                            # FIX C: state=failed 但队列没 error 事件时, 主动 yield
                            # 这是个兜底分支, 正常 fail_task 路径应该会先发 error event
                            error_msg = task.error or "任务执行失败"
                            yield {
                                "event": "error",
                                "data": json.dumps(
                                    {"message": error_msg, "task_id": task_id_str},
                                    ensure_ascii=False,
                                ),
                            }
                        elif task.state.value == "cancelled":
                            # 取消也算终端状态, 显式 yield done (不再有 done 时)
                            pass  # 下面的 done 会处理
                        else:
                            # 罕见: task 状态不是 completed/failed/cancelled 但循环退出了
                            # 防止前端 stuck thinking
                            yield {
                                "event": "error",
                                "data": json.dumps(
                                    {"message": f"任务异常终止, 状态: {task.state.value}"},
                                    ensure_ascii=False,
                                ),
                            }

                    yield {
                        "event": "done",
                        "data": json.dumps(
                            {"task_id": task_id_str, "state": task.state.value},
                            ensure_ascii=False,
                        ),
                    }
                    break

                et = event.event_type

                # 业务轨过滤：仅 audience='observability' 的事件不下发
                # thinking_step / reasoning 默认 audience='both' → 业务轨下发，
                # 用于 StageTimeline 实时反馈（前端在 useAgentStore handler 内累积）
                if _should_skip_in_business_track(event):
                    if et == "thinking_step":
                        seen_step_count += 1
                    continue

                if et == "result":
                    yield {
                        "event": "result",
                        "data": json.dumps(
                            _transform_result_for_frontend(event.data),
                            ensure_ascii=False,
                        ),
                    }
                    yield {
                        "event": "done",
                        "data": json.dumps(
                            {"task_id": task_id_str, "state": "completed"},
                            ensure_ascii=False,
                        ),
                    }
                    break
                elif et == "error":
                    yield {
                        "event": "error",
                        "data": json.dumps(event.data, ensure_ascii=False),
                    }
                    yield {
                        "event": "done",
                        "data": json.dumps(
                            {"task_id": task_id_str, "state": "failed"},
                            ensure_ascii=False,
                        ),
                    }
                    break
                else:
                    # plan / stage / progress / llm_chunk / task_created /
                    # thinking_step / reasoning —— 业务轨可见的内部状态事件
                    yield {
                        "event": et,
                        "data": json.dumps(event.data, ensure_ascii=False),
                    }
        except Exception as e:
            logger.exception("SSE v2 流错误", _type="system", event_name="sse_v2_error",
                             message=f"SSE v2 流错误: {e}", status="error")
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}, ensure_ascii=False),
            }
        finally:
            if task_id_str:
                event_bus.unsubscribe(task_id_str, on_event)

    return EventSourceResponse(event_generator())


# ===================== 业务 SSE v3（spec 2026-07-02） =====================


@router.post("/agent/stream/v3")
async def agent_stream_v3(
    request: AgentStreamRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user),
):
    """业务轨 SSE v3 — Parts 模型（spec 2026-07-02 §3.3）

    关键变更：
    - PartWriter 内存缓冲 + 200ms 节流刷 DB
    - 9 种 SSE 事件类型（spec §3.3）
    - eventBus 原始事件在 endpoint 层翻译为 v3 事件并写入 PartWriter
    - 一次性切流：上线后 /agent/stream/v2 下线
    """
    from .part_writer import PartWriter
    from src.database import SessionLocal
    import uuid as _uuid

    http_trace_id = http_request.headers.get("X-Trace-Id", "")
    task_id_str = ""

    async def event_generator() -> AsyncGenerator[dict, None]:
        nonlocal task_id_str

        # 初始化（复用 v2 逻辑）
        skill_context, user_prompt = _init_stream_context(
            request=request,
            current_user=current_user,
            http_trace_id=http_trace_id,
        )

        # 创建 PartWriter
        db = SessionLocal()
        part_writer: PartWriter | None = None
        try:
            message_id = str(_uuid.uuid4())
            part_writer = PartWriter(message_id, db)

            # message_start
            yield {
                "event": "message_start",
                "data": json.dumps({"message_id": message_id, "role": "assistant"}, ensure_ascii=False),
            }

            # plan 占位
            yield {
                "event": "plan",
                "data": json.dumps({
                    "part_id": f"plan_{int(time.time()*1000)}",
                    "goal": user_prompt[:200],
                    "reasoning": "ManagerAgent 正在动态规划...",
                    "tasks": [],
                }, ensure_ascii=False),
            }

            # EventBus 订阅
            evt_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

            async def on_event(event: StreamEvent) -> None:
                await evt_queue.put(event)

            task_id_str = task_queue.reserve_task_id()
            event_bus.subscribe(task_id_str, on_event)

            await task_queue.enqueue(
                prompt=user_prompt,
                user_id=current_user.id,
                plan=None,
                skill_context=skill_context if skill_context else None,
                task_id=task_id_str,
            )

            yield {
                "event": "task_created",
                "data": json.dumps({"task_id": task_id_str, "message_id": message_id}, ensure_ascii=False),
            }

            timeout_seconds = 600
            start_time = asyncio.get_event_loop().time()

            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout_seconds:
                    yield {
                        "event": "error",
                        "data": json.dumps({"code": "TIMEOUT", "message": "任务执行超时"}, ensure_ascii=False),
                    }
                    break

                try:
                    event = await asyncio.wait_for(evt_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    task = task_queue.get_task(task_id_str)
                    if not task or task.state.value in ("completed", "failed", "cancelled"):
                        break
                    continue

                et = event.event_type
                data = event.data or {}

                # ── 翻译为 v3 事件 + 写入 PartWriter ──────────────────
                if et == "plan":
                    part_id = f"plan_{int(time.time()*1000)}"
                    yield {
                        "event": "message_start",
                        "data": json.dumps({"message_id": message_id, "role": "assistant"}, ensure_ascii=False),
                    }
                    yield {
                        "event": "plan",
                        "data": json.dumps({
                            "part_id": part_id,
                            "goal": data.get("goal", ""),
                            "reasoning": data.get("reasoning", ""),
                            "tasks": data.get("tasks", []),
                        }, ensure_ascii=False),
                    }
                    if part_writer:
                        await part_writer.append("plan", part_id, {
                            "goal": data.get("goal", ""),
                            "reasoning": data.get("reasoning", ""),
                            "tasks": data.get("tasks", []),
                        })

                elif et == "stage":
                    agent = data.get("agent", "")
                    label = data.get("label", "")
                    icon = data.get("icon", "⚙️")
                    status = data.get("status", "running")
                    round_ = data.get("reviewRound", 0)
                    duration_ms = data.get("duration_ms")
                    error_msg = data.get("error")
                    part_id = data.get("part_id") or f"tool_{agent}_{round_}_{int(time.time()*1000)}"

                    if status == "running" and part_writer:
                        await part_writer.append_tool_call(part_id, agent, label, icon, round_)

                    yield {
                        "event": "stage",
                        "data": json.dumps({
                            "part_id": part_id,
                            "agent": agent,
                            "label": label,
                            "icon": icon,
                            "status": status,
                            "round": round_,
                            **({"duration_ms": duration_ms} if duration_ms else {}),
                            **({"error": error_msg} if error_msg else {}),
                        }, ensure_ascii=False),
                    }

                    if status in ("done", "completed", "error") and part_writer:
                        state = "error" if error_msg else "done"
                        await part_writer.update(part_id, {
                            "state": state,
                            "output": data.get("output"),
                            "duration_ms": duration_ms,
                            "error_message": error_msg,
                        })
                        yield {
                            "event": "tool_result",
                            "data": json.dumps({
                                "part_id": part_id,
                                "output": data.get("output"),
                                "duration_ms": duration_ms,
                                "error": error_msg,
                            }, ensure_ascii=False),
                        }

                elif et == "llm_chunk":
                    delta = data.get("chunk") or data.get("delta") or data.get("content") or ""
                    if delta and part_writer:
                        await part_writer.append_text("text_main", delta)
                    yield {
                        "event": "text_delta",
                        "data": json.dumps({"part_id": "text_main", "delta": delta}, ensure_ascii=False),
                    }

                elif et == "reasoning":
                    agent = data.get("agent", "")
                    round_ = data.get("round")
                    content = data.get("content") or data.get("delta") or ""
                    sid_suffix = f"{agent}" + (f"_r{round_}" if round_ else "")
                    part_id = f"reasoning_{sid_suffix}_{int(time.time()*1000)}"
                    if content and part_writer:
                        await part_writer.append_reasoning(
                            part_id, content,
                            {"agent": agent, "round": round_} if round_ else {"agent": agent},
                        )
                    yield {
                        "event": "reasoning_delta",
                        "data": json.dumps({
                            "part_id": part_id,
                            "agent": agent,
                            "delta": content,
                            **({"round": round_} if round_ else {}),
                        }, ensure_ascii=False),
                    }

                elif et == "result":
                    output = data.get("output", {})
                    full_text = output.get("full_text") or output.get("content", "")
                    if part_writer:
                        await part_writer.append("artifact", "artifact_main", {
                            "artifact_type": _infer_artifact_type(output),
                            "body": full_text,
                            "title": output.get("title"),
                            "platform": output.get("platform"),
                            "structured": output.get("structured"),
                        })
                    yield {
                        "event": "artifact",
                        "data": json.dumps({
                            "part_id": "artifact_main",
                            "artifact_type": _infer_artifact_type(output),
                            "body": full_text,
                            "title": output.get("title"),
                            "platform": output.get("platform"),
                            "structured": output.get("structured"),
                        }, ensure_ascii=False),
                    }
                    if part_writer:
                        await part_writer.finalize()
                    task = task_queue.get_task(task_id_str)
                    usage: dict = {}
                    if task and task.result:
                        res = task.result or {}
                        usage = {
                            "tokens_used": res.get("usage", {}).get("tokens_used"),
                            "time_elapsed": res.get("usage", {}).get("time_elapsed"),
                            "model": res.get("usage", {}).get("model"),
                        }
                    yield {
                        "event": "message_end",
                        "data": json.dumps({
                            "message_id": message_id,
                            "status": "completed",
                            "usage": usage,
                        }, ensure_ascii=False),
                    }
                    break

                elif et == "error":
                    if part_writer:
                        await part_writer.finalize()
                    yield {
                        "event": "error",
                        "data": json.dumps({
                            "code": "AGENT_ERROR",
                            "message": data.get("message", str(data)),
                        }, ensure_ascii=False),
                    }
                    yield {
                        "event": "message_end",
                        "data": json.dumps({"message_id": message_id, "status": "failed"}, ensure_ascii=False),
                    }
                    break

        except Exception as e:
            logger.exception("SSE v3 流错误", _type="system", event_name="sse_v3_error",
                           message=f"SSE v3 流错误: {e}", status="error")
            yield {
                "event": "error",
                "data": json.dumps({"code": "INTERNAL_ERROR", "message": str(e)}, ensure_ascii=False),
            }
        finally:
            if task_id_str:
                event_bus.unsubscribe(task_id_str, on_event)
            if part_writer:
                try:
                    await part_writer.finalize()
                except Exception:
                    pass
            db.close()

    return EventSourceResponse(event_generator())


# ===================== Observability SSE（spec 2026-07-01） =====================
# 只供 dev panel / 后端调试使用。下发所有内部状态事件（thinking_step / reasoning + 全量 result dump）
# 默认关闭，未来通过 frontend dev panel 或 admin tool 订阅


@router.get("/agent/obs/stream/{task_id}")
async def agent_obs_stream(
    task_id: str,
    current_user: User = Depends(get_current_user),
):
    """Observability 轨 SSE（spec 2026-07-01）

    行为：
    - 订阅指定 task_id 的 EventBus
    - 客户端通过 query param `?token=<jwt>` 鉴权（与 WebSocket 一致）
    - 下发所有内部状态事件：thinking_step / reasoning / stage / progress / 全量 result dump
    - 业务轨客户端不应使用此端点

    注意：本端点是 OBSERVABILITY-ONLY，不会下发 frontend-facing 业务事件
    """
    # 简化：复用 task_queue 的"subscribe + drain"模式
    # 注意：当前实现是基于 ORCHESTRATOR 提交后的 task_id，不是 SSE 连接时的 task_id

    async def event_generator() -> AsyncGenerator[dict, None]:
        task = task_queue.get_task(task_id)
        if not task:
            yield {"event": "error", "data": json.dumps({"message": "task not found"})}
            return

        if task.user_id != current_user.id:
            yield {"event": "error", "data": json.dumps({"message": "forbidden"})}
            return

        # 简单实现：轮询 task，直到结束，再发送 result dump
        yield {"event": "obs_start", "data": json.dumps({"task_id": task_id})}
        while True:
            await asyncio.sleep(1.0)
            task = task_queue.get_task(task_id)
            if not task:
                yield {"event": "error", "data": json.dumps({"message": "task disappeared"})}
                return

            yield {"event": "progress", "data": json.dumps(task.to_dict())}

            if task.state.value in ("completed", "failed", "cancelled"):
                if task.state.value == "completed" and task.result:
                    # 全量 dump（不做收敛）
                    yield {"event": "result", "data": json.dumps(task.result, ensure_ascii=False)}
                yield {"event": "done", "data": json.dumps({"state": task.state.value})}
                return

    return EventSourceResponse(event_generator())


# ─────────────────────────────────────────────────────────
# Stream 共用初始化 + 业务轨过滤（spec 2026-07-01）
# ─────────────────────────────────────────────────────────

# thinking_step / reasoning 现在是 frontend-可见事件（用于 StageTimeline 实时反馈），
# 不再无脑按 type 过滤。过滤改为按 audience 字段判断：
#   - audience='observability' → 业务轨不下发（仅观测轨需要）
#   - audience='frontend' / 'both' → 业务轨下发
# _INTERNAL_EVENT_TYPES_V2 保留作为 **可选** 黑名单（如未来需要屏蔽某个 type 不下发），
# 但目前为空集——所有非 observability 事件都下发。
_INTERNAL_EVENT_TYPES_V2: frozenset = frozenset()


def _should_skip_in_business_track(event: StreamEvent) -> bool:
    """业务轨过滤：仅 audience='observability' 的事件不下发。

    历史注释（已过时）：
        "thinking_step / reasoning 不下发（前端 store 自累积）"
    ——该假设错误：前端没有自累积路径（useAgentStore 对这两个事件的 handler 是 no-op），
    实际导致 StageTimeline / MessageBubble 在流式过程中永远看不到思考过程。
    修复：把"是否下发"的判断改为按 audience 字段，thinking_step / reasoning 默认
    audience='both' → 业务轨下发，前端累积到 stages[].reasoning 与 thinkingSteps。
    """
    # 黑名单兜底（目前空）
    if event.event_type in _INTERNAL_EVENT_TYPES_V2:
        return True
    # 显式 audience='observability' 的事件跳过
    if event.audience == "observability":
        return True
    return False


def _init_stream_context(
    *,
    request: "AgentStreamRequest",
    current_user: "User",
    http_trace_id: str,
) -> tuple[dict, str]:
    """业务轨 SSE 共用的初始化逻辑（Skill + trigger + 平台）"""
    skill_context: dict = {}

    if request.skill_id:
        from src.api.skills import load_skill_full_for_user
        from src.database import SessionLocal
        db = SessionLocal()
        try:
            skill_data = load_skill_full_for_user(request.skill_id, db, current_user.id)
            if skill_data:
                skill_context = {
                    "skill_id": request.skill_id,
                    "skill_name": skill_data.get("name", ""),
                    "skill_system_prompt": skill_data.get("system_prompt", ""),
                    "skill_preset_plan": skill_data.get("preset_plan"),
                    "skill_preset_complexity": skill_data.get("preset_plan", {}).get("complexity", "medium"),
                    "skill_user_template": skill_data.get("user_prompt_template", ""),
                }
            else:
                logger.warning(
                    "未找到 Skill", _type="system", event_name="skill_not_found",
                    message=f"找不到 Skill: {request.skill_id}", status="error",
                )
        finally:
            db.close()

    user_prompt, platform_code = request.prompt, request.platform
    if skill_context.get("skill_user_template"):
        template = skill_context["skill_user_template"]
        template = template.replace("{topic}", user_prompt.split("\n")[0])
        template = template.replace("{content}", user_prompt)
        user_prompt = template

    if request.parameters:
        for k, v in request.parameters.items():
            if v is not None:
                skill_context[k] = v

    if request.trigger and isinstance(request.trigger, dict):
        for k in ("platform", "style", "length", "tone", "audience",
                  "keywords", "revision_focus", "action", "topic"):
            v = request.trigger.get(k)
            if v not in (None, "", [], {}):
                skill_context[k] = v
        trigger_topic = (request.trigger.get("topic") or "").strip()
        if trigger_topic and 3 <= len(trigger_topic) <= 500:
            user_prompt = trigger_topic
        elif request.trigger.get("_prompt"):
            user_prompt = request.trigger["_prompt"]
        if request.trigger.get("_fallback"):
            skill_context["_trigger_fallback"] = True

    if not platform_code:
        from .platform_registry import detect_platform
        platform_code = detect_platform(user_prompt)
    if platform_code:
        skill_context["platform"] = platform_code
        from .platform_profile import get_profile
        profile = get_profile(platform_code)
        if profile:
            skill_context["platform_name"] = profile.get("name", platform_code)
            skill_context["platform_system_prompt"] = profile.get("system_prompt", "")
            skill_context["platform_writing_rules"] = profile.get("writing_rules", {})
            skill_context["platform_agent_pipeline"] = profile.get("agent_pipeline", {})
            skill_context["platform_quality_check"] = profile.get("quality_check", "")

    if http_trace_id:
        skill_context["trace_id"] = http_trace_id

    return skill_context, user_prompt


# ===================== URL 输入处理 =====================


class URLCrawlRequest(BaseModel):
    url: str
    max_length: int = 3000
    analyze: bool = False


@router.post("/agent/crawl", response_model=AgentRunResponse)
async def crawl_url(
    request: URLCrawlRequest,
    current_user: User = Depends(get_current_user),
):
    """抓取并分析指定 URL 的内容

    如果 analyze=True，会进一步用 Agent 对抓取结果进行分析总结。
    适用于长文阅读、竞品分析等场景。
    """
    try:
        safe_url = validate_public_http_url(request.url)
        from .handlers.web_search import handle_url_crawl
        crawl_result = await handle_url_crawl({
            "url": safe_url,
            "max_length": request.max_length,
        }, llm=None)

        if crawl_result.get("status") == "failed":
            return AgentRunResponse(success=False, error=crawl_result.get("error", "抓取失败"))

        if not request.analyze:
            return AgentRunResponse(success=True, data=crawl_result)

        orchestrator = _create_orchestrator()
        content_text = crawl_result.get("output", {}).get("content", "")
        result = await orchestrator.run(
            user_request=f"分析以下从 {safe_url} 抓取的内容，提取核心观点和关键信息：\n\n{content_text[:8000]}",
            context={"force_template": "simple", "user_id": current_user.id},
            timeout=120.0,
        )
        return AgentRunResponse(
            success=result.status.value in ("completed", "partial"),
            data={"crawl": crawl_result, "analysis": result.to_dict()},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("crawl_url 失败", exc_info=True,
                     _type="system", event_name="crawl_error", message=f"crawl_url 失败: {e}", status="error")
        raise HTTPException(status_code=500, detail=str(e))


# ===================== 死信队列管理 =====================


@router.get("/agent/dlq", response_model=AgentRunResponse)
async def list_dlq_tasks(
    current_user: User = Depends(get_current_user),
):
    """查看死信队列中的失败任务列表"""
    tasks = [task for task in task_queue.get_dlq_tasks() if task.user_id == current_user.id]
    return AgentRunResponse(
        success=True,
        data={
            "tasks": [t.to_dict() for t in tasks],
            "total": len(tasks),
        },
    )


@router.post("/agent/dlq/{task_id}/retry", response_model=AgentRunResponse)
async def retry_dlq_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
):
    """从死信队列中重试指定任务"""
    ensure_task_owner(task_queue._dead_letter_queue.get(task_id), current_user)
    task = await task_queue.retry_from_dlq(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务未在死信队列中找到")
    return AgentRunResponse(
        success=True,
        data={"task_id": task.id, "state": task.state.value},
    )


# ===================== VectorSearch 接口 =====================


class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    texts: Optional[list[str]] = None


@router.post("/agent/vector-search", response_model=AgentRunResponse)
async def search_vector_db(
    request: VectorSearchRequest,
    current_user: User = Depends(get_current_user),
):
    """在向量知识库中搜索相似内容"""
    namespace = f"user:{current_user.id}"
    if request.texts:
        vector_search.add_documents(
            texts=request.texts,
            metadatas=[{"user_id": current_user.id, "namespace": namespace}] * len(request.texts),
        )
    result = vector_search.search(
        query=request.query,
        top_k=request.top_k,
        namespace=namespace,
        metadata_filter={"user_id": current_user.id},
    )
    return AgentRunResponse(success=True, data=result)


# ===================== Cache 统计 =====================


@router.get("/agent/cache/stats", response_model=AgentRunResponse)
async def get_cache_stats(
    current_user: User = Depends(get_current_user),
):
    """查看 Agent 缓存统计信息"""
    from .cache import agent_cache, llm_cache
    return AgentRunResponse(success=True, data={
        "agent_cache": agent_cache.stats,
        "llm_cache": llm_cache.stats,
    })


@router.post("/agent/cache/invalidate", response_model=AgentRunResponse)
async def invalidate_cache(
    agent_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
):
    """清空 Agent 缓存"""
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="当前版本已禁用全局缓存失效接口",
    )


@router.get("/agent/metrics")
async def agent_metrics(
    current_user: User = Depends(get_current_user),
):
    """Agent 系统监控指标（StatsCollector 实时数据 + DB 持久化累计）"""
    settings = get_settings()
    from src.observability.metrics import stats_collector
    from src.database import SessionLocal
    from sqlalchemy import text

    active_tasks = sum(
        1 for t in task_queue._tasks.values()
        if t.user_id == current_user.id and t.state.value in ("pending", "running")
    )

    # 实时内存数据（StatsCollector 聚合，同时写入 Prometheus）
    mem = stats_collector.get_stats()

    # DB 持久化累计数据
    db_totals: dict[str, Any] = {
        "agent_runs_total": 0,
        "llm_tokens_total": 0,
        "llm_cost_total": 0,
        "by_model": [],
        "by_provider": [],
        "agents": [],
    }
    try:
        db = SessionLocal()
        try:
            row = db.execute(text("""
                SELECT
                    COUNT(*) as total_runs,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(total_cost), 0) as total_cost,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_count
                FROM agent_tasks
                WHERE user_id = :user_id
            """), {"user_id": current_user.id}).one()
            db_totals["agent_runs_total"] = row.total_runs or 0
            db_totals["llm_tokens_total"] = row.total_tokens or 0
            db_totals["llm_cost_total"] = round(row.total_cost or 0, 4)

            # agent_subtasks: 按 agent_type 统计
            agent_rows = db.execute(text("""
                SELECT
                    s.agent_type,
                    COUNT(*) as calls,
                    COALESCE(SUM(s.tokens_used), 0) as tokens,
                    COALESCE(AVG(s.time_elapsed), 0) as avg_time
                FROM agent_subtasks s
                JOIN agent_tasks t ON t.id = s.task_id
                WHERE t.user_id = :user_id
                  AND s.agent_type IS NOT NULL AND s.agent_type != ''
                GROUP BY s.agent_type
                ORDER BY calls DESC
            """), {"user_id": current_user.id}).fetchall()
            for r in agent_rows:
                db_totals["agents"].append({
                    "name": r.agent_type,
                    "calls": r.calls,
                    "tokens": r.tokens,
                    "avg_time_s": round(r.avg_time, 2),
                })
        finally:
            db.close()
    except Exception:
        pass

    zhipu_key = getattr(settings, 'zhipu_api_key', None) or getattr(settings, 'zhipu_api_key', '')
    qwen_key = getattr(settings, 'qwen_api_key', None) or getattr(settings, 'qwen_api_key', '')
    openai_key = getattr(settings, 'openai_api_key', None) or getattr(settings, 'openai_api_key', '')
    deepseek_key = getattr(settings, 'deepseek_api_key', None) or getattr(settings, 'deepseek_api_key', '')

    return {
        "success": True,
        "data": {
            "cache": {
                "agent_cache": agent_cache.stats,
                "llm_cache": llm_cache.stats,
            },
            "tasks": {
                "active": active_tasks,
                "dlq_size": sum(1 for t in task_queue._dead_letter_queue.values() if t.user_id == current_user.id),
            },
            "worker": {
                "running": worker._running,
            },
            "config": {
                "debug": settings.debug,
                "log_level": settings.log_level,
                "default_llm_provider": settings.default_llm_provider,
                "worker_max_retries": settings.worker_max_retries,
                "mcp_enabled": settings.mcp_enabled,
                "llm_gateway_url": getattr(settings, 'llm_gateway_url', ''),
                "providers": {
                    "zhipu": bool(zhipu_key),
                    "qwen": bool(qwen_key),
                    "openai": bool(openai_key),
                    "deepseek": bool(deepseek_key),
                },
            },
            # ── 实时运行统计（StatsCollector，当前进程）──
            "agent_runs": mem["agent_runs"],
            "llm_calls": mem["llm_calls"],
            "tool_calls": mem["tool_calls"],
            # ── 持久化历史累计（DB 全量）──
            "history": db_totals,
        },
    }

# ===================== WebSocket =====================


@router.websocket("/agent/ws/{task_id}")
async def agent_websocket(websocket: WebSocket, task_id: str):
    """WebSocket ??????????????????"""
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)
        return

    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        await websocket.close(code=1008)
        return

    from src.database import SessionLocal

    db = SessionLocal()
    try:
        current_user = db.query(User).filter(User.id == payload.get("sub")).first()
    finally:
        db.close()

    if not current_user:
        await websocket.close(code=1008)
        return

    task = task_queue.get_task(task_id)
    if task is None or task.user_id != current_user.id:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        while True:
            task = task_queue.get_task(task_id)
            if task is None:
                await websocket.send_json({
                    "type": "error",
                    "message": "?????",
                })
                break

            await websocket.send_json({
                "type": "progress",
                "data": task.to_dict(),
            })

            if task.state.value in ("completed", "failed", "cancelled"):
                if task.state.value == "completed" and task.result:
                    await websocket.send_json({
                        "type": "result",
                        "data": task.result,
                    })
                break

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(
            "WebSocket ?????",
            exc_info=True,
            task_id=task_id,
            _type="websocket",
            event_name="websocket_error",
        )
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            logger.warning("????? WebSocket ??????", exc_info=True)
    finally:
        try:
            await websocket.close()
        except Exception:
            logger.warning("????? WebSocket ??", exc_info=True)
