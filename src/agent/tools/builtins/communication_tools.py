"""通信工具 — 将 MessageBus 方法暴露为 AgentTool

依赖：MessageBus 实例（由 factory 创建时注入）
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from ..base import AgentTool, AgentToolMeta

logger = logging.getLogger(__name__)


def create_communication_tools(message_bus: Any, agent_type: str = "") -> list[AgentTool]:
    """创建所有通信相关工具

    Args:
        message_bus: MessageBus 实例（由 factory 传入）
        agent_type: 当前 Agent 类型（用于 mailbox 感知，默认为空时使用全局 mailbox）
    """
    # 闭包捕获的 agent_type，可在 _check_messages 中使用
    _current_agent = agent_type

    async def _send_message(
        to_agent: str,
        content: str,
        msg_type: str = "request",
        conversation_id: str = "",
    ) -> str:
        """发送消息给其他 Agent

        Args:
            to_agent: 目标 Agent 类型
            content: 消息内容（纯文本或 JSON 字符串）
            msg_type: 消息类型（request/response/broadcast/review_request/suggestion）
            conversation_id: 对话 ID（留空自动生成）
        """
        if message_bus is None:
            return "错误：MessageBus 未启用。"

        try:
            from ..communication.models import Message, MessageType

            type_map = {
                "request": MessageType.REQUEST,
                "response": MessageType.RESPONSE,
                "review_request": MessageType.REVIEW_REQUEST,
                "review_response": MessageType.REVIEW_RESPONSE,
                "suggestion": MessageType.SUGGESTION,
                "notify": MessageType.NOTIFY,
                "query": MessageType.QUERY,
            }
            resolved_type = type_map.get(msg_type, MessageType.REQUEST)
            conv_id = conversation_id or str(uuid.uuid4())

            msg = Message(
                msg_id=str(uuid.uuid4()),
                from_agent="",
                to_agent=to_agent,
                msg_type=resolved_type,
                content={"text": content},
                conversation_id=conv_id,
            )
            msg_id = await message_bus.send(msg)
            return f"消息已发送至 {to_agent}（msg_id={msg_id}，conversation_id={conv_id}）"
        except Exception as e:
            logger.warning("send_message 失败: %s", e)
            return f"消息发送失败: {e}"

    async def _check_messages() -> str:
        """检查当前是否有其他 Agent 发来的待处理消息"""
        if message_bus is None:
            return "错误：MessageBus 未启用。"

        try:
            pending = message_bus.get_pending(_current_agent)
            if not pending:
                return "无待处理消息。"
            lines = [f"你有 {len(pending)} 条待处理消息："]
            for msg in pending:
                lines.append(
                    f"  - [{msg.msg_type.value}] 来自 {msg.from_agent}: "
                    f"{str(msg.content)[:200]}"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.warning("check_messages 失败: %s", e)
            return f"消息检查失败: {e}"

    async def _get_conversation(conversation_id: str) -> str:
        """获取指定对话的历史记录

        Args:
            conversation_id: 对话 ID
        """
        if message_bus is None:
            return "错误：MessageBus 未启用。"

        try:
            history = message_bus.get_conversation(conversation_id)
            if not history:
                return f"未找到对话 {conversation_id}。"
            lines = [f"对话 {conversation_id} 共 {len(history)} 条消息："]
            for msg in history:
                lines.append(
                    f"  [{msg.created_at}] {msg.from_agent} → {msg.to_agent}: "
                    f"{str(msg.content)[:200]}"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.warning("get_conversation 失败: %s", e)
            return f"对话获取失败: {e}"

    async def _peer_call(
        target_agent: str,
        content: str,
        context: str = "",
    ) -> str:
        """向对等 Agent 发起协作请求并等待回复

        Args:
            target_agent: 目标 Agent 名称（如 WritingAgent / EditorAgent）
            content: 请求内容（描述你需要什么帮助）
            context: 额外的上下文信息（可选）
        """
        if message_bus is None:
            return "错误：MessageBus 未启用，无法进行对等协作。"

        try:
            from ..communication.models import Message, MessageType

            msg_id = str(uuid.uuid4())
            msg = Message(
                msg_id=msg_id,
                from_agent=_current_agent,
                to_agent=target_agent,
                msg_type=MessageType.REQUEST,
                content={"text": content, "context": context},
                conversation_id=f"peer-{msg_id[:8]}",
            )
            reply = await message_bus.send_and_wait(msg, timeout=30.0)
            if reply is None:
                return f"错误：{target_agent} 未在 30.0s 内回复。"
            reply_content = reply.content.get("text", "") if isinstance(reply.content, dict) else str(reply.content)
            return f"[{target_agent} 回复]: {reply_content}"
        except Exception as e:
            return f"对等协作失败: {e}"

    return [
        AgentTool(
            name="send_message",
            description="向其他 Agent 发送消息（支持 request/review_request/suggestion 等类型）",
            execute_func=_send_message,
            meta=AgentToolMeta(
                category="communication",
                agent_types=["WritingAgent", "SEOAgent", "ReviewAgent", "ManagerAgent"],
                builtin=True,
            ),
        ),
        AgentTool(
            name="check_messages",
            description="检查是否有其他 Agent 发来的待处理消息",
            execute_func=_check_messages,
            meta=AgentToolMeta(
                category="communication",
                agent_types=["WritingAgent", "SEOAgent", "ReviewAgent"],
                builtin=True,
            ),
        ),
        AgentTool(
            name="get_conversation",
            description="获取指定对话的历史消息记录",
            execute_func=_get_conversation,
            meta=AgentToolMeta(
                category="communication",
                agent_types=["ManagerAgent"],
                builtin=True,
            ),
        ),
        AgentTool(
            name="peer_call",
            description="向对等 Agent 发起协作请求并等待回复（如 WritingAgent→EditorAgent 润色、EditorAgent→ResearcherAgent 事实核查）",
            execute_func=_peer_call,
            meta=AgentToolMeta(
                category="communication",
                agent_types=["WritingAgent", "EditorAgent", "ReviewAgent", "SEOAgent"],
                builtin=True,
            ),
        ),
    ]
