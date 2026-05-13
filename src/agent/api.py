from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError
from typing import Optional, AsyncGenerator
import asyncio
import json
import uuid

from sse_starlette.sse import EventSourceResponse

from .schemas import AgentTask
from .factory import create_orchestrator
from .orchestrator import Orchestrator
from .workflow.task_queue import task_queue
from .workflow.worker import worker
from src.api.auth import get_current_user
from src.models import User

router = APIRouter(tags=["agent"])


class AgentRunRequest(BaseModel):
    prompt: str
    context: Optional[dict] = None


class AgentRunResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


class AgentAsyncSubmitRequest(BaseModel):
    prompt: str


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
        orchestrator = _create_orchestrator()
        result = await orchestrator.run(
            user_request=request.prompt,
            context=request.context or {},
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
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@router.post("/agent/plan", response_model=AgentRunResponse)
async def preview_plan(
    request: AgentRunRequest,
    current_user: User = Depends(get_current_user),
):
    """预览 Agent 执行计划（不实际执行，只展示拆解结果）"""
    try:
        orchestrator = _create_orchestrator()
        plan = await orchestrator._decompose(
            user_request=request.prompt,
            context=request.context or {},
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
}

REPURPOSE_FORMAT_GUIDE = {
    "social_post": "将其转化为 3-5 条适合社交媒体的短帖，每条100-200字，带互动引导和emoji",
    "newsletter": "将其改写为一封邮件Newsletter，包含友好的问候、摘要和行动号召",
    "diagram": "从内容中提取关键概念，生成一份Mermaid.js知识架构图",
    "copywriting": "将其改写为营销文案，突出价值主张和行动号召",
    "script": "将其改写为短视频口播脚本，包含开场钩子和分镜描述",
    "report": "将其扩展为一份结构化的分析报告，包含数据摘要和结论建议",
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
                        logger.warning(f"Platform adapt failed for {platform_code}: {e}")
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
    async_task = await task_queue.enqueue(
        prompt=request.prompt,
        user_id=current_user.id,
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
        raise HTTPException(status_code=404, detail="Task not found")
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
        raise HTTPException(status_code=404, detail="Task not found")
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
        raise HTTPException(status_code=404, detail="Task not found")
    if task.state.value != "awaiting_input":
        raise HTTPException(status_code=400, detail="Task is not awaiting input")

    success = task_queue.submit_decision(
        task_id=task_id,
        decision_id=request.decision_id,
        choice=request.choice,
    )
    if not success:
        raise HTTPException(status_code=400, detail="Decision rejected")

    return {
        "success": True,
        "data": {
            "task_id": task_id,
            "state": "running",
            "message": "Decision accepted, task resumed",
        },
    }


# ===================== SSE 流式接口 =====================


class AgentStreamRequest(BaseModel):
    prompt: str
    context: Optional[dict] = None


@router.post("/agent/stream")
async def agent_stream(
    request: AgentStreamRequest,
    current_user: User = Depends(get_current_user),
):
    """流式 Agent 执行 — SSE 推送计划、思考步骤、进度和结果"""

    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            # 1. 生成执行计划
            orchestrator = _create_orchestrator()
            plan = await orchestrator._decompose(
                user_request=request.prompt,
                context=request.context or {},
            )
            yield {
                "event": "plan",
                "data": json.dumps({
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
                }, ensure_ascii=False),
            }

            # 2. 提交异步任务执行
            async_task = await task_queue.enqueue(
                prompt=request.prompt,
                user_id=current_user.id,
            )

            yield {
                "event": "task_created",
                "data": json.dumps({"task_id": async_task.id}, ensure_ascii=False),
            }

            # 3. 轮询进度并推送
            last_step_count = 0
            last_progress = -1
            timeout_seconds = 300
            start_time = asyncio.get_event_loop().time()

            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout_seconds:
                    yield {
                        "event": "error",
                        "data": json.dumps({"message": "任务执行超时"}, ensure_ascii=False),
                    }
                    break

                task = task_queue.get_task(async_task.id)
                if not task:
                    yield {
                        "event": "error",
                        "data": json.dumps({"message": "任务丢失"}, ensure_ascii=False),
                    }
                    break

                # 推送进度
                if task.progress != last_progress:
                    last_progress = task.progress
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "progress": task.progress,
                            "current_step": task.current_step,
                            "progress_message": task.progress_message,
                            "completed_steps": task.completed_steps,
                            "total_steps": task.total_steps,
                        }, ensure_ascii=False),
                    }

                # 推送新增的 thinking steps
                if task.result and isinstance(task.result, dict):
                    trace = task.result.get("thinking_trace", {})
                    steps = trace.get("steps", []) if isinstance(trace, dict) else []
                    if len(steps) > last_step_count:
                        for step in steps[last_step_count:]:
                            yield {
                                "event": "thinking_step",
                                "data": json.dumps(step, ensure_ascii=False),
                            }
                        last_step_count = len(steps)

                # 检查是否需要用户决策
                if task.state.value == "awaiting_input" and task.pending_decision:
                    pd = task.pending_decision
                    yield {
                        "event": "decision_required",
                        "data": json.dumps({
                            "task_id": async_task.id,
                            "decision_id": pd.get("decision_id"),
                            "question": pd.get("question", ""),
                            "options": pd.get("options", []),
                            "context": pd.get("context", {}),
                        }, ensure_ascii=False),
                    }
                    break

                # 任务完成
                if task.state.value in ("completed", "failed", "cancelled"):
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
                        "data": json.dumps({"task_id": async_task.id, "state": task.state.value}, ensure_ascii=False),
                    }
                    break

                await asyncio.sleep(0.5)

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


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
                    "message": "Task not found",
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
