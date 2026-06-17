"""对话端点 — SSE 流式多轮对话

提供 /api/v1/conversation/chat 端点，前端直接调用。
对话 LLM 自然交流，在信息足够时输出 [GENERATE] 标记让前端触发生成。
"""

from __future__ import annotations
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session

from src.database import get_db
from src.api.auth import get_current_user
from src.models import User, ChatConversation, ChatMessageRecord
from src.conversation import ConversationEngine
from src.memory.manager import memory_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/conversation", tags=["conversation"])


# ===================== 请求/响应模型 =====================


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户消息")
    conversation_id: Optional[str] = Field(None, description="对话 ID，首次不传则自动创建")


# ===================== 对话端点 =====================


@router.post("/chat")
async def conversation_chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """对话聊天端点 — SSE 流式回复

    输入用户消息，返回 SSE 流：
    - type: text → 对话文本片段（流式）
    - type: text_cleanup → 移除 [GENERATE] 标记后的完整文本
    - type: trigger → 生成触发指令（前端收到后调 /agent/stream）

    如果未传 conversation_id，自动创建新对话。
    """
    conversation_id = request.conversation_id

    # 自动创建对话
    if not conversation_id:
        conv = ChatConversation(
            user_id=current_user.id,
            title=request.message[:30] + ("..." if len(request.message) > 30 else ""),
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conversation_id = conv.id

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
    from datetime import datetime
    conv = db.query(ChatConversation).filter(ChatConversation.id == conversation_id).first()
    if conv:
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

            # 2. 加载用户偏好记忆
            user_profile = await memory_manager.get_user_profile(current_user.id)

            # 3. 构建对话上下文
            from src.conversation.schemas import ConversationContext
            ctx = ConversationContext(
                message=request.message,
                conversation_id=conversation_id,
                user_id=current_user.id,
                history=history,
                user_profile=user_profile,
            )

            # 4. 调用对话引擎
            engine = ConversationEngine()
            trigger_data = None
            accumulated_text = ""

            async for event in engine.chat(ctx):
                if event["type"] == "text":
                    accumulated_text += event["content"]
                    yield {
                        "event": "text",
                        "data": json.dumps({"content": event["content"]}, ensure_ascii=False),
                    }
                elif event["type"] == "text_cleanup":
                    accumulated_text = event["content"]
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
                # 更新 assistant 消息的 result_json
                asst_msg.result_json = json.dumps({
                    "trigger": trigger_data,
                    "type": trigger_data.get("action", "create"),
                }, ensure_ascii=False)
                db.commit()

                # 记住用户偏好
                await memory_manager.remember_conversation_preference(
                    user_id=current_user.id,
                    preferences=trigger_data,
                )

                # 返回 trigger 事件给前端
                yield {
                    "event": "trigger",
                    "data": json.dumps(trigger_data, ensure_ascii=False),
                }

            # 返回 done 事件
            yield {
                "event": "done",
                "data": json.dumps({
                    "conversation_id": conversation_id,
                    "message_id": asst_msg.id if accumulated_text else None,
                }, ensure_ascii=False),
            }

        except Exception as e:
            logger.exception("对话引擎出错")
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())
