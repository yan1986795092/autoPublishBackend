from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError
from typing import Optional, AsyncGenerator
import asyncio
import json
import logging
import uuid

logger = logging.getLogger(__name__)

# ===================== SSE 流式接口 =====================


from sse_starlette.sse import EventSourceResponse

from .schemas import AgentTask
from .factory import create_orchestrator
from .orchestrator import Orchestrator
from .workflow.task_queue import task_queue, event_bus, StreamEvent
from .workflow.worker import worker
from .cache import agent_cache, llm_cache
from src.api.auth import get_current_user
from src.models import User
from src.tools.vector_search import vector_search
from src.tools.langchain import CrawlerTool
from src.config import get_settings

router = APIRouter(tags=["agent"])


class AgentRunRequest(BaseModel):
    prompt: str
    context: Optional[dict] = None
    skill_id: Optional[str] = None  # 应用指定的 Skill ID


class AgentRunResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


class AgentAsyncSubmitRequest(BaseModel):
    prompt: str
    skill_id: Optional[str] = None


class AgentAsyncStatusResponse(BaseModel):
    success: bool
    data: Optional[dict] = None


class DecisionSubmitRequest(BaseModel):
    decision_id: str
    choice: str
    text: Optional[str] = None


def _create_orchestrator(**kwargs) -> Orchestrator:
    """创建 Orchestrator 实例（通过 ModelRouter 为各 Agent 分配独立 LLM 配置）"""
    return create_orchestrator(**kwargs)


# ===================== 同步接口 =====================


@router.post("/agent/run", response_model=AgentRunResponse)
async def run_agent(
    request: AgentRunRequest,
    current_user: User = Depends(get_current_user),
):
    """运行 Agent 编排：输入需求，返回 Multi-Agent 协作结果"""
    try:
        context = request.context or {}

        # Skill 注入（与 /stream 端点保持一致）
        if request.skill_id:
            from src.api.skills import load_skill_full
            from src.database import SessionLocal
            db = SessionLocal()
            try:
                skill_data = load_skill_full(request.skill_id, db)
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

        # Skill 注入
        if request.skill_id:
            from src.api.skills import load_skill_full
            from src.database import SessionLocal
            db = SessionLocal()
            try:
                skill_data = load_skill_full(request.skill_id, db)
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
                                    "task": f"Adapt the following content for {platform_code} platform format",
                                },
                            ),
                            context={},
                            previous_results={},
                        )
                        platform_outputs[platform_code] = platform_task.output.get("content", output_content)
                    except Exception as e:
                        logger.warning(f"平台适配失败 {platform_code}: {e}")
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
        from src.api.skills import load_skill_full
        from src.database import SessionLocal
        db = SessionLocal()
        try:
            skill_data = load_skill_full(request.skill_id, db)
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
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="未找到任务")
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
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="未找到任务")
    success = task_queue.cancel_task(task_id)
    return {
        "success": success,
        "data": {"task_id": task_id, "state": "cancelled"},
    }


# ===================== 用户决策接口 =====================


@router.post("/agent/decision/{task_id}")
async def submit_decision(
    task_id: str,
    request: DecisionSubmitRequest,
    current_user: User = Depends(get_current_user),
):
    """提交用户决策，让等待中的 Agent 任务继续执行"""
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="未找到任务")
    if task.state.value != "awaiting_input":
        raise HTTPException(status_code=400, detail="任务不在等待输入状态")

    success = task_queue.submit_decision(
        task_id=task_id,
        decision_id=request.decision_id,
        choice=request.choice,
    )
    if not success:
        raise HTTPException(status_code=400, detail="决策被拒绝")

    return {
        "success": True,
        "data": {
            "task_id": task_id,
            "state": "running",
            "message": "决策已接受，任务已恢复",
        },
    }


# ===================== SSE 流式接口 =====================


class AgentStreamRequest(BaseModel):
    prompt: str
    context: Optional[dict] = None
    skill_id: Optional[str] = None  # 应用指定的 Skill ID


@router.post("/agent/stream")
async def agent_stream(
    request: AgentStreamRequest,
    current_user: User = Depends(get_current_user),
):
    """流式 Agent 执行 — 通过 EventBus 实时推送计划、思考步骤、进度和结果

    核心改进：不再轮询 TaskQueue，改为订阅 EventBus，事件驱动的实时推送。
    支持事件类型：plan, task_created, thinking_step, llm_chunk, reasoning, progress, result, error, done
    """
    task_id_str = ""  # 实际 task_id 在入队后设置

    async def event_generator() -> AsyncGenerator[dict, None]:
        nonlocal task_id_str
        try:
            # 0. 加载 Skill（如果指定了 skill_id）
            skill_context: dict = {}
            if request.skill_id:
                from src.api.skills import load_skill_full
                from src.database import SessionLocal
                db = SessionLocal()
                try:
                    skill_data = load_skill_full(request.skill_id, db)
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
                        logger.warning(f"未找到 Skill: {request.skill_id}")
                finally:
                    db.close()

            # 应用 Skill 的 user_prompt_template 转换
            user_prompt = request.prompt
            if skill_context.get("skill_user_template"):
                template = skill_context["skill_user_template"]
                template = template.replace("{topic}", user_prompt.split("\n")[0])
                template = template.replace("{content}", user_prompt)
                user_prompt = template

            # 1. 生成执行计划（支持 Skill preset_plan + 平台感知覆盖）
            orchestrator = _create_orchestrator()

            # 优先使用 skill preset_plan
            if skill_context.get("skill_preset_plan"):
                preset = skill_context["skill_preset_plan"]
                plan = orchestrator._build_template_plan(
                    user_request=user_prompt,
                    complexity=preset.get("complexity", "medium"),
                )
            else:
                # 平台感知：检测平台类型，使用平台注册表
                platform_code = orchestrator._detect_platform(user_prompt)
                if platform_code:
                    plan = orchestrator._build_platform_plan(user_prompt, platform_code)
                else:
                    plan = orchestrator._build_default_plan(user_prompt)
            yield {
                "event": "plan",
                "data": json.dumps({
                    "goal": plan.goal,
                    "reasoning": plan.reasoning,
                    "estimated_steps": plan.estimated_steps,
                    "tasks": [
                        {
                            "id": t.id,
                            "agent_type": t.agent_type,
                            "input": t.input,
                            "dependencies": t.dependencies,
                        }
                        for t in plan.tasks
                    ],
                }, ensure_ascii=False),
            }

            # 2. 创建 EventBus 订阅队列（线程安全队列用于跨协程传递事件）
            evt_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

            async def on_event(event: StreamEvent) -> None:
                """EventBus 回调：将事件放入队列"""
                await evt_queue.put(event)

            # 3. 提交异步任务（传递预计算计划 + Skill 上下文）
            plan_dict = {
                "goal": plan.goal,
                "reasoning": plan.reasoning,
                "estimated_steps": plan.estimated_steps,
                "tasks": [
                    {
                        "id": t.id,
                        "agent_type": t.agent_type,
                        "input": t.input,
                        "dependencies": t.dependencies,
                        "config": {},
                    }
                    for t in plan.tasks
                ],
            }
            async_task = await task_queue.enqueue(
                prompt=user_prompt,
                user_id=current_user.id,
                plan=plan_dict,
                skill_context=skill_context if skill_context else None,
            )
            task_id_str = async_task.id

            # 4. 订阅 EventBus（在 enqueue 之后，使用正确的 task_id）
            event_bus.subscribe(task_id_str, on_event)

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
                    elif et == "decision_required":
                        task = task_queue.get_task(task_id_str)
                        if task and task.pending_decision:
                            pd = task.pending_decision
                            yield {
                                "event": "decision_required",
                                "data": json.dumps({
                                    "task_id": task_id_str,
                                    "decision_id": pd.get("decision_id"),
                                    "question": pd.get("question", ""),
                                    "options": pd.get("options", []),
                                    "context": pd.get("context", {}),
                                }, ensure_ascii=False),
                            }
                        break
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
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


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
        from .handlers.web_search import handle_url_crawl
        crawl_result = await handle_url_crawl({
            "url": request.url,
            "max_length": request.max_length,
        }, llm=None)

        if crawl_result.get("status") == "failed":
            return AgentRunResponse(success=False, error=crawl_result.get("error", "抓取失败"))

        if not request.analyze:
            return AgentRunResponse(success=True, data=crawl_result)

        orchestrator = _create_orchestrator()
        content_text = crawl_result.get("output", {}).get("content", "")
        result = await orchestrator.run(
            user_request=f"分析以下从 {request.url} 抓取的内容，提取核心观点和关键信息：\n\n{content_text[:8000]}",
            context={"force_template": "simple"},
            timeout=120.0,
        )
        return AgentRunResponse(
            success=result.status.value in ("completed", "partial"),
            data={"crawl": crawl_result, "analysis": result.to_dict()},
        )
    except Exception as e:
        logger.error(f"crawl_url 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ===================== 死信队列管理 =====================


@router.get("/agent/dlq", response_model=AgentRunResponse)
async def list_dlq_tasks(
    current_user: User = Depends(get_current_user),
):
    """查看死信队列中的失败任务列表"""
    tasks = task_queue.get_dlq_tasks()
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
    task = task_queue.retry_from_dlq(task_id)
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
    if request.texts:
        vector_search.add_documents(
            texts=request.texts,
            metadatas=[{"user_id": current_user.id}] * len(request.texts),
        )
    result = vector_search.search(query=request.query, top_k=request.top_k)
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
    from .cache import agent_cache, llm_cache
    agent_cache.invalidate(agent_type)
    if not agent_type:
        llm_cache.invalidate()
    return AgentRunResponse(success=True, data={"message": "缓存已清空"})


@router.get("/agent/metrics")
async def agent_metrics():
    """Agent 系统监控指标"""
    settings = get_settings()

    active_tasks = sum(
        1 for t in task_queue._tasks.values()
        if t.state.value in ("pending", "running", "awaiting_input")
    )

    return {
        "success": True,
        "data": {
            "cache": {
                "agent_cache": agent_cache.stats,
                "llm_cache": llm_cache.stats,
            },
            "tasks": {
                "active": active_tasks,
                "dlq_size": len(task_queue._dead_letter_queue),
            },
            "worker": {
                "running": worker._running,
            },
            "config": {
                "debug": settings.debug,
                "log_level": settings.log_level,
                "default_llm_provider": settings.default_llm_provider,
                "enable_reflection": settings.enable_reflection,
                "enable_message_bus": settings.enable_message_bus,
                "enable_hitl": settings.enable_hitl,
                "worker_max_retries": settings.worker_max_retries,
            },
        },
    }

# ===================== WebSocket =====================


@router.websocket("/agent/ws/{task_id}")
async def agent_websocket(websocket: WebSocket, task_id: str):
    """WebSocket 端点：实时推送 Agent 任务进度 + 用户决策请求"""
    await websocket.accept()
    try:
        while True:
            task = task_queue.get_task(task_id)
            if task is None:
                await websocket.send_json({
                    "type": "error",
                    "message": "未找到任务",
                })
                break

            task_dict = task.to_dict()
            await websocket.send_json({
                "type": "progress",
                "data": task_dict,
            })

            # 如果需要用户决策，发送 decision_required 消息
            if task.state.value == "awaiting_input" and task.pending_decision:
                pd = task.pending_decision
                await websocket.send_json({
                    "type": "decision_required",
                    "data": {
                        "task_id": task_id,
                        "decision_id": pd.get("decision_id"),
                        "question": pd.get("question", "需要你的决定"),
                        "options": pd.get("options", []),
                        "context": pd.get("context", {}),
                    },
                })

            if task.state.value in ("completed", "failed", "cancelled"):
                if task.state.value == "completed" and task.result:
                    await websocket.send_json({
                        "type": "result",
                        "data": task.result,
                    })
                break

            # 等待下一次更新（轮询）
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
