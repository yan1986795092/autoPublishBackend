"""MessageBus — Agent 间点对点消息总线

两种通信模式：
1. 直接消息: Agent A → Agent B (send/wait_reply)
2. 发布订阅: Agent A → 所有订阅者 (publish/subscribe)

消息按 conversation_id 分组，支持回复链追踪。
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Optional, Callable, Awaitable

from .models import Message, MessageType

logger = logging.getLogger(__name__)

# 消息回调类型
MessageHandler = Callable[[Message], Awaitable[None]]


class MessageBus:
    """Agent 间消息总线

    非线程安全，单线程 async 使用。
    每个 Orchestrator 实例创建独立的 MessageBus。
    """

    def __init__(self, registry: Any = None):
        self._registry = registry                     # AgentRegistry（可选）
        self._outbox: dict[str, list[Message]] = {}   # agent_type -> [pending messages]
        self._history: dict[str, list[Message]] = {}  # conversation_id -> [messages]
        self._subscriptions: dict[str, list[MessageHandler]] = {}  # agent_type -> [handlers]
        self._reply_events: dict[str, asyncio.Event] = {}         # msg_id -> event
        self._replies: dict[str, Message] = {}                     # in_reply_to -> Message

    # ── 发送消息 ──

    async def send(self, msg: Message) -> str:
        """发送消息到目标 Agent 的 outbox"""
        self._append_history(msg)
        if msg.to_agent:
            self._ensure_outbox(msg.to_agent).append(msg)
        logger.debug("MessageBus: %s → %s [%s]", msg.from_agent, msg.to_agent, msg.msg_type.value)
        return msg.msg_id

    async def reply(self, original: Message, content: dict, from_agent: str) -> Message:
        """回复消息"""
        target = original.from_agent if original.from_agent != from_agent else original.to_agent
        msg = Message(
            msg_id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=target or original.from_agent,
            msg_type=MessageType.RESPONSE,
            content=content,
            conversation_id=original.conversation_id,
            in_reply_to=original.msg_id,
        )
        await self.send(msg)

        # 通知等待 reply 的协程
        if original.msg_id in self._reply_events:
            self._replies[original.msg_id] = msg
            self._reply_events[original.msg_id].set()

        return msg

    async def send_and_wait(
        self,
        msg: Message,
        timeout: float = 30.0,
    ) -> Optional[Message]:
        """发送消息并等待回复"""
        await self.send(msg)
        event = asyncio.Event()
        self._reply_events[msg.msg_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self._replies.pop(msg.msg_id, None)
        except asyncio.TimeoutError:
            logger.warning("MessageBus: reply timeout for msg=%s", msg.msg_id)
            return None
        finally:
            self._reply_events.pop(msg.msg_id, None)
            self._replies.pop(msg.msg_id, None)

    async def broadcast(self, msg: Message) -> None:
        """广播消息给所有接入的 Agent"""
        msg.to_agent = None
        self._append_history(msg)
        # 广播消息进入每个 Agent 的 outbox
        for agent_type in self._outbox:
            if agent_type != msg.from_agent:
                self._outbox[agent_type].append(msg)
        logger.debug("MessageBus: %s broadcast [%s]", msg.from_agent, msg.msg_type.value)

    # ── 接收消息 ──

    def get_pending(self, agent_type: str) -> list[Message]:
        """获取并清空 Agent 的待处理消息"""
        return self._outbox.pop(agent_type, [])

    def peek_pending(self, agent_type: str) -> list[Message]:
        """查看待处理消息（不清空）"""
        return self._outbox.get(agent_type, [])

    def has_pending(self, agent_type: str) -> bool:
        """检查是否有待处理消息"""
        msgs = self._outbox.get(agent_type, [])
        return len(msgs) > 0

    # ── 对话历史 ──

    def get_conversation(self, conversation_id: str) -> list[Message]:
        """获取对话历史"""
        return self._history.get(conversation_id, [])

    def get_all_conversations(self) -> dict[str, list[Message]]:
        return dict(self._history)

    def get_messages_between(
        self, agent_a: str, agent_b: str,
    ) -> list[Message]:
        """获取两个 Agent 之间的所有消息"""
        result = []
        for conv in self._history.values():
            for msg in conv:
                if {msg.from_agent, msg.to_agent} == {agent_a, agent_b}:
                    result.append(msg)
        return result

    # ── 订阅 ──

    def subscribe(self, agent_type: str, handler: MessageHandler) -> None:
        """订阅某个 Agent 类型的消息事件"""
        if agent_type not in self._subscriptions:
            self._subscriptions[agent_type] = []
        self._subscriptions[agent_type].append(handler)

    async def notify_subscribers(self, msg: Message) -> None:
        """通知订阅者"""
        handlers = self._subscriptions.get(msg.to_agent or "", [])
        for handler in handlers:
            try:
                await handler(msg)
            except Exception as e:
                logger.warning("MessageBus: subscriber error: %s", e)

    # ── 内部 ──

    def _ensure_outbox(self, agent_type: str) -> list[Message]:
        if agent_type not in self._outbox:
            self._outbox[agent_type] = []
        return self._outbox[agent_type]

    def _append_history(self, msg: Message) -> None:
        conv_id = msg.conversation_id or "_default"
        if conv_id not in self._history:
            self._history[conv_id] = []
        self._history[conv_id].append(msg)

    def clear(self) -> None:
        """清空所有状态（每次 run 前调用）"""
        self._outbox.clear()
        self._history.clear()
        self._reply_events.clear()
        self._replies.clear()

    # ── Agent-initiated communication ──

    def agent_discovery_sync(self, requester: str, capability_needed: str) -> Optional[str]:
        """Synchronous version of agent discovery."""
        if self._registry:
            matches = self._registry.find_by_expertise(capability_needed)
            if matches:
                logger.debug("Agent discovery: %s needs %s → %s", requester, capability_needed, matches[0].name)
                return matches[0].name
        return None

    async def agent_discovery(
        self,
        requester: str,
        capability_needed: str,
    ) -> Optional[str]:
        """Ask the agent registry who can handle a capability. Returns agent_type or None."""
        return self.agent_discovery_sync(requester, capability_needed)

    async def peer_feedback(
        self,
        from_agent: str,
        to_agent: str,
        feedback_content: dict,
        related_task_id: Optional[str] = None,
    ) -> None:
        """
        Send direct feedback from one agent to another (e.g., ReviewAgent → WritingAgent).
        Fire-and-forget: does NOT wait for a reply.
        """
        msg = Message(
            msg_id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            msg_type=MessageType.SUGGESTION,
            content=feedback_content,
            conversation_id=f"peer-{from_agent}-{to_agent}",
            metadata={"related_task_id": related_task_id} if related_task_id else {},
        )
        self._append_history(msg)
        self._ensure_outbox(to_agent).append(msg)
        logger.debug("Peer feedback: %s → %s [%s]", from_agent, to_agent, feedback_content.get("type", "unknown"))

    def __repr__(self) -> str:
        total_msgs = sum(len(v) for v in self._history.values())
        return f"MessageBus({total_msgs} msgs, {len(self._outbox)} outboxes)"


