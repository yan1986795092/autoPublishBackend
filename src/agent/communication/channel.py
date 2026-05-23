"""ConversationChannel — 多 Agent 对话通道

支持多个 Agent 在共享频道中持续对话。
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from .models import Message, MessageType
from .bus import MessageBus

logger = logging.getLogger(__name__)


class ConversationChannel:
    """多 Agent 对话通道"""

    def __init__(
        self,
        channel_id: str,
        participants: list[str],
        bus: MessageBus,
        topic: str = "",
    ):
        self.channel_id = channel_id
        self.participants = list(participants)
        self._bus = bus
        self.topic = topic

    # ── 消息发送 ──

    async def broadcast(self, content: dict, sender: str) -> list[Message]:
        """向频道内所有其他参与者发送消息"""
        sent = []
        for participant in self.participants:
            if participant != sender:
                msg = Message(
                    msg_id=str(uuid.uuid4()),
                    from_agent=sender,
                    to_agent=participant,
                    msg_type=MessageType.BROADCAST,
                    content=content,
                    conversation_id=self.channel_id,
                    metadata={"channel": self.channel_id, "topic": self.topic},
                )
                await self._bus.send(msg)
                sent.append(msg)
        return sent

    async def send_to(self, sender: str, recipient: str, content: dict) -> Message:
        """向频道内特定参与者发送消息"""
        msg = Message(
            msg_id=str(uuid.uuid4()),
            from_agent=sender,
            to_agent=recipient,
            msg_type=MessageType.REQUEST,
            content=content,
            conversation_id=self.channel_id,
            metadata={"channel": self.channel_id, "topic": self.topic},
        )
        await self._bus.send(msg)
        return msg

    # ── 对话历史 ──

    def get_history(self) -> list[Message]:
        """获取频道内的完整对话历史"""
        return self._bus.get_conversation(self.channel_id)

    def get_history_for(self, participant: str) -> list[Message]:
        """获取特定参与者的消息"""
        return [
            msg for msg in self.get_history()
            if msg.from_agent == participant or msg.to_agent == participant
        ]

    # ── 生命周期 ──

    def add_participant(self, agent_type: str) -> None:
        if agent_type not in self.participants:
            self.participants.append(agent_type)

    def remove_participant(self, agent_type: str) -> None:
        if agent_type in self.participants:
            self.participants.remove(agent_type)

    @property
    def message_count(self) -> int:
        return len(self.get_history())
