"""Agent 通信数据模型"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class MessageType(str, Enum):
    """消息类型"""
    REQUEST = "request"                 # 请求
    RESPONSE = "response"               # 响应
    BROADCAST = "broadcast"             # 广播
    REVIEW_REQUEST = "review_request"   # 审阅请求
    REVIEW_RESPONSE = "review_response" # 审阅回复
    SUGGESTION = "suggestion"           # 建议
    QUERY = "query"                     # 查询
    NOTIFY = "notify"                   # 通知
    NEGOTIATE = "negotiate"             # 协商
    CHANNEL_JOIN = "channel_join"       # 加入频道
    CHANNEL_LEAVE = "channel_leave"     # 离开频道


@dataclass
class Message:
    """Agent 间消息"""
    msg_id: str
    from_agent: str
    to_agent: Optional[str]          # None = broadcast
    msg_type: MessageType
    content: dict
    conversation_id: str
    in_reply_to: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "msg_id": self.msg_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "msg_type": self.msg_type.value if isinstance(self.msg_type, MessageType) else self.msg_type,
            "content": self.content,
            "conversation_id": self.conversation_id,
            "in_reply_to": self.in_reply_to,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
