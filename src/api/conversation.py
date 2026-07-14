"""对话端点 — SSE 流式多轮对话

提供 /api/v1/conversation/chat 端点，前端直接调用。
对话 LLM 自然交流，在信息足够时输出 [GENERATE] 标记让前端触发生成。
"""

from __future__ import annotations
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session

from src.database import get_db
from src.api.auth import get_current_user
from src.models import User, ChatConversation, ChatMessageRecord
from src.conversation import ConversationEngine
from src.memory.manager import memory_manager
from src.core.request_guards import ensure_conversation_owner
from src.agent.stream_core import run_agent_sse_generator
from src.agent.api import _init_stream_context, AgentStreamRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/conversation", tags=["conversation"])


# ===================== 辅助函数 =====================


def _inject_revision_source(
    trigger_data: dict,
    conversation_id: str,
    db: Session,
) -> None:
    """revise trigger 缺少 content/draft_content 时，从对话历史自动找最近正文"""
    if trigger_data.get("content") or trigger_data.get("draft_content"):
        return  # 已有来源，无需注入

    try:
        # 查最近一条包含实际 agent 输出的 assistant 消息（result_json 含 output 字段，
        # 而非仅 trigger 元信息），跳过当前刚生成的 trigger 消息自身
        candidates = (
            db.query(ChatMessageRecord)
            .filter(
                ChatMessageRecord.conversation_id == conversation_id,
                ChatMessageRecord.role == "assistant",
                ChatMessageRecord.result_json.isnot(None),
            )
            .order_by(ChatMessageRecord.created_at.desc())
            .limit(10)
            .all()
        )
        for msg in candidates:
            if not msg.result_json:
                continue
            try:
                result = json.loads(msg.result_json)
                # 跳过只有 trigger 的消息（对话引擎的 [GENERATE] 标记消息）
                if not isinstance(result, dict) or "output" not in result:
                    continue
                output = result.get("output") or {}
                full_text = output.get("full_text") or output.get("content") or ""
                draft_text = output.get("draft_content") or ""
                if full_text:
                    trigger_data["content"] = full_text
                    if draft_text:
                        trigger_data["draft_content"] = draft_text
                    logger.info(
                        "revise source auto-injected (%d chars) from conversation %s",
                        len(full_text), conversation_id,
                    )
                    return
            except json.JSONDecodeError:
                continue
    except Exception as e:
        logger.warning("revise source injection failed: %s", e)


# ===================== 请求/响应模型 =====================


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户消息")
    conversation_id: Optional[str] = Field(None, description="对话 ID，首次不传则自动创建")
    platform: Optional[str] = Field(None, description="用户已选的发布平台（如 xiaohongshu）")
    content_type: Optional[str] = Field(None, description="用户已选的写作体裁（如 copywriting）")
    preferences: Optional[dict] = Field(
        None,
        description="前端生成参数（creativity/length/tone/audience），由对话 LLM 透传",
    )


# ===================== 对话端点 =====================


@router.post("/chat")
async def conversation_chat(
    request: ChatRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """对话聊天端点 — SSE 流式回复（v4 直连 Agent Pipeline）

    输入用户消息，返回 SSE 流：
    - event: text → 对话文本片段（流式）
    - event: text_cleanup → 移除标记后的完整文本
    - event: message_start → Agent Pipeline 开始
    - event: text_delta / stage / reasoning_delta / artifact → 生成事件
    - event: message_end / done → 结束

    如果未传 conversation_id，自动创建新对话。
    """
    conversation_id = request.conversation_id

    # 自动创建对话（修复 #9：同用户、同标题、24h 内复用，避免重复创建 N 条同名对话）
    if not conversation_id:
        from datetime import datetime, timedelta
        new_title = request.message[:30] + ("..." if len(request.message) > 30 else "")
        cutoff = datetime.utcnow() - timedelta(hours=24)
        existing = (
            db.query(ChatConversation)
            .filter(
                ChatConversation.user_id == current_user.id,
                ChatConversation.title == new_title,
                ChatConversation.updated_at >= cutoff,
            )
            .order_by(ChatConversation.updated_at.desc())
            .first()
        )
        if existing:
            conversation_id = existing.id
            existing.updated_at = datetime.utcnow()
            db.commit()
        else:
            conv = ChatConversation(
                user_id=current_user.id,
                title=new_title,
            )
            db.add(conv)
            db.commit()
            db.refresh(conv)
            conversation_id = conv.id
    else:
        conv = ensure_conversation_owner(
            db.query(ChatConversation).filter(
                ChatConversation.id == conversation_id,
                ChatConversation.user_id == current_user.id,
            ).first(),
            current_user,
        )

    # 保存用户消息
    user_msg = ChatMessageRecord(
        conversation_id=conversation_id,
        role="user",
        content=request.message,
        status="completed",
    )
    db.add(user_msg)
    db.flush()

    # 更新对话时间
    from datetime import datetime  # noqa: F811 — used later in scope
    conv.updated_at = datetime.utcnow()

    db.commit()

    async def event_generator():
        try:
            # 1. 加载对话历史
            history_records = (
                db.query(ChatMessageRecord)
                .filter(
                    ChatMessageRecord.conversation_id == conversation_id,
                    ChatMessageRecord.id != user_msg.id,  # 排除刚插入的
                )
                .order_by(ChatMessageRecord.created_at.asc())
                .all()
            )

            history = [
                {"role": msg.role, "content": msg.content}
                for msg in history_records
            ]

            # 3. 构建对话上下文
            from src.conversation.schemas import ConversationContext
            ctx = ConversationContext(
                message=request.message,
                conversation_id=conversation_id,
                user_id=current_user.id,
                history=history,
                user_profile="",
                platform=request.platform or "",
                content_type=request.content_type or "",
                preferences=request.preferences or {},
            )

            # 4. 调用对话引擎
            engine = ConversationEngine()
            trigger_data = None
            accumulated_text = ""

            async for event in engine.chat(ctx):
                if event["type"] == "text":
                    accumulated_text += event["content"]
                    # 流式透传 text 事件，让前端逐 token 渲染
                    # 即使后面有 trigger，text_cleanup 会覆盖 accumulatedText，
                    # 短暂显示原始文本（含 [GENERATE] 标记）是可接受的
                    yield {
                        "event": "text",
                        "data": json.dumps({"content": event["content"]}, ensure_ascii=False),
                    }
                elif event["type"] == "text_cleanup":
                    accumulated_text = event["content"]
                    # text_cleanup 只在有 trigger 时发生，此时 yield 清理后的文本替换前端内容
                    yield {
                        "event": "text_cleanup",
                        "data": json.dumps({"content": event["content"]}, ensure_ascii=False),
                    }
                elif event["type"] == "trigger":
                    trigger_data = event["data"]
                    # 不在这里 yield trigger，等文本流结束后再 yield

            # 5. 保存对话 LLM 的回复
            if accumulated_text:
                asst_msg = ChatMessageRecord(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=accumulated_text,
                    status="completed",
                )
                db.add(asst_msg)
                db.commit()

            # 6. 如果有 trigger，保存到 assistant 消息 result 中 + yield
            if trigger_data:
                # 6a. revise 时自动注入最近正文/草稿作为修订来源
                if trigger_data.get("action") == "revise":
                    _inject_revision_source(trigger_data, conversation_id, db)

                # 更新 assistant 消息的 result_json
                asst_msg.result_json = json.dumps({
                    "trigger": trigger_data,
                    "type": trigger_data.get("action", "create"),
                }, ensure_ascii=False)
                try:
                    db.commit()
                except Exception as commit_err:
                    logger.warning("result_json 持久化失败，不阻塞 trigger 推送: %s", commit_err)

                # 记住用户偏好（按 conversation_id 隔离）
                # 注意：此方法可能抛异常（如 Qwen Embedding 初始化失败），
                # 必须 try/except 包裹，不能阻塞后续 trigger 推送。
                try:
                    await memory_manager.remember_conversation_preference(
                        user_id=current_user.id,
                        preferences=trigger_data,
                        conversation_id=conversation_id,
                    )
                    logger.info("对话偏好已记录，准备推送 trigger 事件到前端")
                except Exception as mem_err:
                    logger.warning("记录对话偏好失败，不阻塞 trigger 推送: %s", mem_err)

                # ── 触发 Agent Pipeline（不再返回 trigger 事件给前端转发）──
                # v4 重构：对话引擎识别到 trigger 后，后端直连 Agent Pipeline，
                # 所有事件（message_start, plan, text_delta, artifact 等）
                # 走同一条 SSE 流推给前端，消除 trigger 事件丢失风险。
                logger.info(
                    "对话引擎触发 Agent Pipeline: action=%s topic=%r",
                    trigger_data.get("action"), str(trigger_data.get("topic", ""))[:60],
                )

                # 提取 trace_id 用于端到端关联
                http_trace_id = http_request.headers.get("X-Trace-Id", "")

                # 构建 AgentStreamRequest 复用 _init_stream_context
                agent_req = AgentStreamRequest(
                    prompt=trigger_data.get("_prompt") or trigger_data.get("topic", ""),
                    platform=trigger_data.get("platform"),
                    parameters=trigger_data.get("parameters"),
                    trigger=trigger_data,
                )
                skill_context, user_prompt = _init_stream_context(
                    request=agent_req,
                    current_user=current_user,
                    http_trace_id=http_trace_id,
                )

                # 直连 Agent Pipeline，通过同一条 SSE 流推送
                # 传入 asst_msg.id 让 parts 写入同一消息（否则 stream_core 会新建消息）
                async for agent_event in run_agent_sse_generator(
                    user_prompt=user_prompt,
                    skill_context=skill_context,
                    current_user=current_user,
                    http_trace_id=http_trace_id,
                    message_id=asst_msg.id,
                    conversation_id=conversation_id,
                ):
                    yield agent_event

            # 返回 done 事件
            yield {
                "event": "done",
                "data": json.dumps({
                    "conversation_id": conversation_id,
                    "message_id": asst_msg.id if accumulated_text else None,
                }, ensure_ascii=False),
            }

        except Exception as e:
            logger.exception("对话引擎出错: %s", e)
            # 安全地尝试发 error 事件
            try:
                yield {
                    "event": "error",
                    "data": json.dumps({"message": str(e)}, ensure_ascii=False),
                }
                # 发一个 done 让前端能正常关闭状态
                yield {
                    "event": "done",
                    "data": json.dumps({
                        "conversation_id": conversation_id,
                        "message_id": None,
                    }, ensure_ascii=False),
                }
            except Exception:
                # yield 本身失败时静默，不二次崩溃
                pass

    return EventSourceResponse(event_generator())
