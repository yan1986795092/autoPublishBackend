"""对话历史持久化 API"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, asc

from src.database import get_db
from src.api.auth import get_current_user
from src.models import User, ChatConversation, ChatMessageRecord
from src.api.chat_message_ui import build_message_ui_fields

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


# ===================== 请求/响应模型 =====================


class MessageCreate(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    status: str = "completed"
    result: Optional[dict] = None
    task_id: Optional[str] = None
    plan: Optional[dict] = None


class MessageUpdate(BaseModel):
    content: Optional[str] = None
    status: Optional[str] = None
    result: Optional[dict] = None


class ConversationCreate(BaseModel):
    title: Optional[str] = None


class ConversationUpdate(BaseModel):
    title: Optional[str] = None


class ConversationPinUpdate(BaseModel):
    """置顶更新请求"""
    is_pinned: bool


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    status: str
    result: Optional[dict] = None
    task_id: Optional[str] = None
    plan: Optional[dict] = None
    kind: Optional[str] = None
    render_mode: Optional[str] = None
    render_payload: Optional[dict] = None
    is_generation: Optional[bool] = None
    is_conversation: Optional[bool] = None
    generation_meta: Optional[dict] = None
    parts: Optional[list] = None  # Parts 模型（spec 2026-07-02 §3.1）
    created_at: str

    class Config:
        from_attributes = True


class ConversationResponse(BaseModel):
    id: str
    title: str
    is_pinned: bool = False
    created_at: str
    updated_at: str
    message_count: int = 0
    last_message: Optional[MessageResponse] = None

    class Config:
        from_attributes = True


def _build_message_response(
    *,
    id: str,
    role: str,
    content: str,
    status: str,
    result: Optional[dict],
    task_id: Optional[str],
    plan: Optional[dict],
    created_at: str,
    parts: Optional[list] = None,
) -> MessageResponse:
    ui = build_message_ui_fields(
        role=role,
        status=status,
        content=content,
        result=result,
        task_id=task_id,
        plan=plan,
    )
    return MessageResponse(
        id=id,
        role=role,
        content=content,
        status=status,
        result=result,
        task_id=task_id,
        plan=plan,
        kind=ui["kind"],
        render_mode=ui["render_mode"],
        render_payload=ui["render_payload"],
        is_generation=ui["is_generation"],
        is_conversation=ui["is_conversation"],
        generation_meta=ui["generation_meta"],
        parts=parts,
        created_at=created_at,
    )


# ===================== 对话 API =====================


@router.get("/conversations")
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
    q: Optional[str] = None,
):
    """获取用户的对话列表（支持标题搜索）

    排序规则：
    1. 置顶对话优先（按 updated_at 降序）
    2. 非置顶对话按 updated_at 降序
    """
    # 显式 user_id 过滤：防止 tenant 隔离失效导致跨用户数据泄露
    # （ORM hook 在某些 FastAPI 执行路径下 ContextVar 传递不可靠）
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    from sqlalchemy import func
    base_filter = ChatConversation.user_id == current_user.id
    query = db.query(ChatConversation).filter(base_filter)
    if q:
        query = query.filter(ChatConversation.title.ilike(f"%{q}%"))

    total = query.count()

    query = query.order_by(
        desc(ChatConversation.is_pinned),
        desc(ChatConversation.updated_at)
    )
    conversations = query.offset(offset).limit(limit).all()

    if not conversations:
        return {
            "success": True,
            "data": {
                "conversations": [],
                "total": total,
                "offset": offset,
                "limit": limit,
            },
        }

    conv_ids = [c.id for c in conversations]

    count_rows = db.query(
        ChatMessageRecord.conversation_id,
        func.count(ChatMessageRecord.id),
    ).filter(
        ChatMessageRecord.conversation_id.in_(conv_ids)
    ).group_by(
        ChatMessageRecord.conversation_id
    ).all()
    msg_counts: dict[str, int] = {cid: cnt for cid, cnt in count_rows}

    latest_msg_subq = (
        db.query(
            ChatMessageRecord,
            func.row_number().over(
                partition_by=ChatMessageRecord.conversation_id,
                order_by=ChatMessageRecord.created_at.desc(),
            ).label("rn"),
        )
        .filter(ChatMessageRecord.conversation_id.in_(conv_ids))
        .subquery()
    )
    latest_msg_alias = db.query(
        latest_msg_subq.c.id,
        latest_msg_subq.c.conversation_id,
        latest_msg_subq.c.role,
        latest_msg_subq.c.content,
        latest_msg_subq.c.status,
        latest_msg_subq.c.result_json,
        latest_msg_subq.c.task_id,
        latest_msg_subq.c.plan_json,
        latest_msg_subq.c.created_at,
    ).filter(latest_msg_subq.c.rn == 1).all()

    last_msgs: dict[str, any] = {row.conversation_id: row for row in latest_msg_alias}

    result = []
    for conv in conversations:
        last_msg = last_msgs.get(conv.id)
        last_msg_response = None
        if last_msg:
            last_result = json.loads(last_msg.result_json) if last_msg.result_json else None
            last_plan = json.loads(last_msg.plan_json) if last_msg.plan_json else None
            last_msg_response = _build_message_response(
                id=last_msg.id,
                role=last_msg.role,
                content=last_msg.content,
                status=last_msg.status,
                result=last_result,
                task_id=last_msg.task_id,
                plan=last_plan,
                created_at=last_msg.created_at.isoformat() if last_msg.created_at else "",
            )

        result.append(ConversationResponse(
            id=conv.id,
            title=conv.title,
            is_pinned=conv.is_pinned,
            created_at=conv.created_at.isoformat() if conv.created_at else "",
            updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
            message_count=msg_counts.get(conv.id, 0),
            last_message=last_msg_response,
        ))

    return {
        "success": True,
        "data": {
            "conversations": result,
            "total": total,
            "offset": offset,
            "limit": limit,
        }
    }


@router.post("/conversations")
async def create_conversation(
    request: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """创建新对话"""
    conversation = ChatConversation(
        user_id=current_user.id,
        title=request.title or "新对话",
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return {
        "success": True,
        "data": ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            is_pinned=conversation.is_pinned,
            created_at=conversation.created_at.isoformat() if conversation.created_at else "",
            updated_at=conversation.updated_at.isoformat() if conversation.updated_at else "",
            message_count=0,
            last_message=None,
        )
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取对话详情（含所有消息）

    返回的消息包含 parts[] 字段（Parts 模型 spec 2026-07-02 §3.1）：
      - 有 DB parts → 返回 parts[]
      - 无 DB parts 且有 result_json → 走迁移 shim
    """
    from src.models import ChatMessagePart
    from src.api.chat_message_ui_v2 import build_message_parts, build_preview_from_parts

    conversation = db.query(ChatConversation).filter(
        ChatConversation.id == conversation_id,
        ChatConversation.user_id == current_user.id,
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")

    messages = db.query(ChatMessageRecord).filter(
        ChatMessageRecord.conversation_id == conversation_id
    ).order_by(ChatMessageRecord.created_at.asc()).all()

    if not messages:
        return {
            "success": True,
            "data": {
                "conversation": ConversationResponse(
                    id=conversation.id,
                    title=conversation.title,
                    is_pinned=conversation.is_pinned,
                    created_at=conversation.created_at.isoformat() if conversation.created_at else "",
                    updated_at=conversation.updated_at.isoformat() if conversation.updated_at else "",
                    message_count=0,
                ),
                "messages": [],
            }
        }

    msg_ids = [m.id for m in messages]

    # 批量查询 parts
    parts_rows = db.query(ChatMessagePart).filter(
        ChatMessagePart.message_id.in_(msg_ids)
    ).order_by(ChatMessagePart.seq.asc()).all()

    # message_id → parts[]
    parts_by_msg: dict[str, list] = {}
    for row in parts_rows:
        parts_by_msg.setdefault(row.message_id, []).append(row)
    # 保持 seq 顺序
    for mid in parts_by_msg:
        parts_by_msg[mid].sort(key=lambda r: r.seq)

    message_responses = []
    for msg in messages:
        db_parts = parts_by_msg.get(msg.id, [])
        msg_parts = build_message_parts(msg.id, db_parts, msg.result_json)

        message_responses.append(_build_message_response(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            status=msg.status,
            result=json.loads(msg.result_json) if msg.result_json else None,
            task_id=msg.task_id,
            plan=json.loads(msg.plan_json) if msg.plan_json else None,
            created_at=msg.created_at.isoformat() if msg.created_at else "",
            parts=msg_parts,
        ))

    return {
        "success": True,
        "data": {
            "conversation": ConversationResponse(
                id=conversation.id,
                title=conversation.title,
                is_pinned=conversation.is_pinned,
                created_at=conversation.created_at.isoformat() if conversation.created_at else "",
                updated_at=conversation.updated_at.isoformat() if conversation.updated_at else "",
                message_count=len(messages),
            ),
            "messages": message_responses,
        }
    }


@router.put("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    request: ConversationUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """更新对话（如标题）"""
    conversation = db.query(ChatConversation).filter(
        ChatConversation.id == conversation_id,
        ChatConversation.user_id == current_user.id,
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")

    if request.title is not None:
        conversation.title = request.title

    # 显式 bump updated_at：解决 SQLAlchemy onupdate 仅在 dirty field 时触发的问题
    conversation.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(conversation)

    return {
        "success": True,
        "data": ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            is_pinned=conversation.is_pinned,
            created_at=conversation.created_at.isoformat() if conversation.created_at else "",
            updated_at=conversation.updated_at.isoformat() if conversation.updated_at else "",
        )
    }


@router.patch("/conversations/{conversation_id}/pin")
async def pin_conversation(
    conversation_id: str,
    request: ConversationPinUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """置顶/取消置顶对话"""
    conversation = db.query(ChatConversation).filter(
        ChatConversation.id == conversation_id,
        ChatConversation.user_id == current_user.id,
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")

    conversation.is_pinned = request.is_pinned
    conversation.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(conversation)

    return {
        "success": True,
        "data": ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            is_pinned=conversation.is_pinned,
            created_at=conversation.created_at.isoformat() if conversation.created_at else "",
            updated_at=conversation.updated_at.isoformat() if conversation.updated_at else "",
        )
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """删除对话（级联删除所有消息）"""
    conversation = db.query(ChatConversation).filter(
        ChatConversation.id == conversation_id,
        ChatConversation.user_id == current_user.id,
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")

    db.delete(conversation)
    db.commit()

    return {"success": True, "data": {"id": conversation_id}}


# ===================== 消息 API =====================


@router.post("/conversations/{conversation_id}/messages")
async def add_message(
    conversation_id: str,
    request: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """添加消息到对话"""
    conversation = db.query(ChatConversation).filter(
        ChatConversation.id == conversation_id,
        ChatConversation.user_id == current_user.id,
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")

    message = ChatMessageRecord(
        conversation_id=conversation_id,
        role=request.role,
        content=request.content,
        status=request.status,
        result_json=json.dumps(request.result) if request.result else None,
        task_id=request.task_id,
        plan_json=json.dumps(request.plan) if request.plan else None,
    )
    db.add(message)

    # 更新对话的 updated_at 时间
    conversation.updated_at = datetime.utcnow()

    # 如果是第一条用户消息且对话标题是默认的，自动设置标题
    # 省略号统一用 …（前端 ConversationHistory 截断展示也用 …）
    if request.role == "user" and conversation.title == "新对话":
        conversation.title = request.content[:30] + ("…" if len(request.content) > 30 else "")

    db.commit()
    db.refresh(message)

    # 修复 BUG-11：add_message 返回值缺少 UI 字段（kind/render_mode/...
    # 会导致前端"addMessage 后立即 updateMessage 之前"的消息渲染模式错误。
    # 用 _build_message_response 统一构建，保证与 list / update_message 返回结构一致。
    result = json.loads(message.result_json) if message.result_json else None
    plan = json.loads(message.plan_json) if message.plan_json else None

    return {
        "success": True,
        "data": _build_message_response(
            id=message.id,
            role=message.role,
            content=message.content,
            status=message.status,
            result=result,
            task_id=message.task_id,
            plan=plan,
            created_at=message.created_at.isoformat() if message.created_at else "",
        )
    }


@router.put("/messages/{message_id}")
async def update_message(
    message_id: str,
    request: MessageUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """更新消息（如更新状态、内容、结果）"""
    message = db.query(ChatMessageRecord).join(ChatConversation).filter(
        ChatMessageRecord.id == message_id,
        ChatConversation.user_id == current_user.id,
    ).first()

    if not message:
        raise HTTPException(status_code=404, detail="消息不存在")

    if request.content is not None:
        message.content = request.content
    if request.status is not None:
        message.status = request.status
    if request.result is not None:
        message.result_json = json.dumps(request.result)

    # bump 对话的 updated_at，让左侧历史能正确按"今天/昨天"分组。
    # add_message 里也会 bump，这里补齐 update_message 路径。
    message.conversation.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(message)

    return {
        "success": True,
        "data": _build_message_response(
            id=message.id,
            role=message.role,
            content=message.content,
            status=message.status,
            result=json.loads(message.result_json) if message.result_json else None,
            task_id=message.task_id,
            plan=json.loads(message.plan_json) if message.plan_json else None,
            created_at=message.created_at.isoformat() if message.created_at else "",
        )
    }


@router.delete("/messages/{message_id}")
async def delete_message(
    message_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """删除单条消息"""
    message = db.query(ChatMessageRecord).join(ChatConversation).filter(
        ChatMessageRecord.id == message_id,
        ChatConversation.user_id == current_user.id,
    ).first()

    if not message:
        raise HTTPException(status_code=404, detail="消息不存在")

    db.delete(message)
    db.commit()

    return {"success": True, "data": {"id": message_id}}


# ===================== 快捷接口：创建对话并添加首条消息 =====================
# DEPRECATED: /api/v1/chat/quick-start 已废弃。
# 新的推荐流程是：
#   1) POST /api/v1/chat/conversations    → 拿 conversation_id（同时同步 UI 列表）
#   2) POST /api/v1/chat/conversations/{id}/messages    → 写首条 user 消息
#   3) POST /api/v1/conversation/chat    → 对话 LLM 流式
# 前端 `useAgentStore.runStreaming` 已于 2026-07-01 切换到上述流程。
# 此接口保留是为了避免线上旧客户端 500，仅在 2026-09-30 前兼容。
# 届时将直接删除，请尽快迁移。


@router.post("/quick-start", deprecated=True)
async def quick_start_conversation(
    request: MessageCreate,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """快速开始：创建对话并添加首条消息，返回 conversation_id

    ⚠️ DEPRECATED — 2026-09-30 下线。请改用：
       POST /api/v1/chat/conversations
       POST /api/v1/chat/conversations/{id}/messages
    """
    logger.warning(
        "DEPRECATED /api/v1/chat/quick-start 被调用，请迁移到 /conversations + /messages 流程",
        extra={"user_id": current_user.id},
    )
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "2026-09-30T00:00:00Z"
    response.headers["Link"] = (
        '</api/v1/chat/conversations>; rel="successor-version"'
    )

    if request.role != "user":
        raise HTTPException(status_code=400, detail="首条消息必须是 user 角色")

    conversation = ChatConversation(
        user_id=current_user.id,
        title=request.content[:30] + ("…" if len(request.content) > 30 else ""),
    )
    db.add(conversation)
    db.flush()  # 获取 conversation.id

    message = ChatMessageRecord(
        conversation_id=conversation.id,
        role=request.role,
        content=request.content,
        status=request.status,
        result_json=json.dumps(request.result) if request.result else None,
        task_id=request.task_id,
        plan_json=json.dumps(request.plan) if request.plan else None,
    )
    db.add(message)
    db.commit()
    db.refresh(conversation)
    db.refresh(message)

    return {
        "success": True,
        "data": {
            "conversation_id": conversation.id,
            "message": _build_message_response(
                id=message.id,
                role=message.role,
                content=message.content,
                status=message.status,
                result=json.loads(message.result_json) if message.result_json else None,
                task_id=message.task_id,
                plan=json.loads(message.plan_json) if message.plan_json else None,
                created_at=message.created_at.isoformat() if message.created_at else "",
            ),
        }
    }
