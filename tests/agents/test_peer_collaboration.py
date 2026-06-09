"""对等协作测试 — 验证 Agent 之间通过 MessageBus 的协作流程"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agent.react.sub_agent import SubAgent
from src.agent.agents.profile import AgentProfile
from src.agent.schemas import SubAgentConfig
from src.agent.communication.models import Message, MessageType


@pytest.fixture
def writer_profile():
    return AgentProfile(
        name="WritingAgent",
        role="写作 Agent",
        description="生成文章",
        expertise=["写作"],
        handler_key="writing",
        tools_enabled=True,
    )


@pytest.fixture
def editor_profile():
    return AgentProfile(
        name="EditorAgent",
        role="编辑 Agent",
        description="润色文章",
        expertise=["编辑", "润色"],
        handler_key="editor",
        tools_enabled=True,
    )


@pytest.fixture
def mock_message_bus():
    bus = MagicMock()
    bus._registry = MagicMock()
    bus._registry.list_types.return_value = ["WritingAgent", "EditorAgent", "ReviewAgent"]
    return bus


class TestPeerCollaboration:
    @pytest.mark.asyncio
    async def test_writer_calls_editor(self, writer_profile, mock_message_bus):
        """WriterAgent 通过 MessageBus 向 EditorAgent 发起 peer_call"""
        mock_llm = AsyncMock()
        response = MagicMock()
        response.content = '{"thought":"需要编辑润色","action":"peer_call","action_input":{"target_agent":"EditorAgent","content":"请润色这段文字：赛博朋克2077是一款赛博朋克风格的角色扮演游戏。"}}'
        mock_llm.ainvoke.return_value = response

        editor_response = MagicMock()
        editor_response.content = {"text": "已润色：赛博朋克2077是一款赛博朋克风格的动作角色扮演游戏，由CDPR开发。"}
        mock_message_bus.send_and_wait = AsyncMock(return_value=editor_response)

        writer = SubAgent(
            profile=writer_profile,
            llm=mock_llm,
            message_bus=mock_message_bus,
            config=SubAgentConfig(max_react_steps=5),
        )
        result = await writer.run_with_react(
            task="介绍赛博朋克2077",
            context=None,
        )

        mock_message_bus.send_and_wait.assert_called()
        call_args = mock_message_bus.send_and_wait.call_args
        msg_sent = call_args[0][0]
        assert msg_sent.to_agent == "EditorAgent"
        assert "润色" in str(msg_sent.content)

    @pytest.mark.asyncio
    async def test_peer_call_timeout(self, writer_profile, mock_message_bus):
        """peer_call 超时时返回失败信息，不阻塞循环"""
        mock_llm = AsyncMock()
        response = MagicMock()
        response.content = '{"thought":"需要编辑","action":"peer_call","action_input":{"target_agent":"EditorAgent","content":"请润色"}}'
        mock_llm.ainvoke.return_value = response

        mock_message_bus.send_and_wait = AsyncMock(return_value=None)

        writer = SubAgent(
            profile=writer_profile,
            llm=mock_llm,
            message_bus=mock_message_bus,
            config=SubAgentConfig(max_react_steps=3, peer_call_timeout=0.1),
        )
        result = await writer.run_with_react(
            task="写一段话",
            context=None,
        )

        assert "output" in result
        failed_calls = [c for c in result.get("peer_calls", []) if not c.get("success")]
        assert len(failed_calls) > 0

    @pytest.mark.asyncio
    async def test_peer_call_exception(self, writer_profile, mock_message_bus):
        """peer_call 抛出异常时返回错误信息"""
        mock_llm = AsyncMock()
        response = MagicMock()
        response.content = '{"thought":"需要协作","action":"peer_call","action_input":{"target_agent":"EditorAgent","content":"请润色"}}'
        mock_llm.ainvoke.return_value = response

        mock_message_bus.send_and_wait = AsyncMock(side_effect=RuntimeError("连接失败"))

        writer = SubAgent(
            profile=writer_profile,
            llm=mock_llm,
            message_bus=mock_message_bus,
            config=SubAgentConfig(max_react_steps=3),
        )
        result = await writer.run_with_react(
            task="写一段话",
            context=None,
        )

        assert "output" in result
        failed_calls = [c for c in result.get("peer_calls", []) if not c.get("success")]
        assert len(failed_calls) > 0
        assert "连接失败" in failed_calls[0]["error"]

    @pytest.mark.asyncio
    async def test_no_peer_call_when_disabled(self, writer_profile, mock_message_bus):
        """对等协作未启用时，不调用 MessageBus"""
        mock_llm = AsyncMock()
        response = MagicMock()
        response.content = '{"thought":"需要编辑","action":"peer_call","action_input":{"target_agent":"EditorAgent","content":"润色"}}'
        mock_llm.ainvoke.return_value = response

        writer = SubAgent(
            profile=writer_profile,
            llm=mock_llm,
            message_bus=mock_message_bus,
            config=SubAgentConfig(enable_peer_collaboration=False),
        )
        result = await writer.run_with_react(
            task="写一段话",
            context=None,
        )

        peer_calls = result.get("peer_calls", [])
        assert len(peer_calls) > 0
        assert not peer_calls[0]["success"]
        mock_message_bus.send_and_wait.assert_not_called()

    @pytest.mark.asyncio
    async def test_chain_collaboration(self, writer_profile, editor_profile, mock_message_bus):
        """验证连续多次 peer_call 的流程"""
        mock_llm = AsyncMock()
        call_count = [0]

        def llm_side_effect(messages):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                response.content = '{"thought":"需要编辑","action":"peer_call","action_input":{"target_agent":"EditorAgent","content":"请润色"}}'
            elif call_count[0] == 2:
                response.content = '{"thought":"编辑完成，任务结束","action":"finish","action_input":{}}'
            else:
                response.content = '{"thought":"任务完成","action":"finish","action_input":{}}'
            return response

        mock_llm.ainvoke = AsyncMock(side_effect=llm_side_effect)

        editor_reply_count = [0]

        def bus_side_effect(msg, timeout=None):
            editor_reply_count[0] += 1
            reply = MagicMock()
            reply.content = {"text": f"回复{editor_reply_count[0]}：已处理"}
            return reply

        mock_message_bus.send_and_wait = AsyncMock(side_effect=bus_side_effect)

        writer = SubAgent(
            profile=writer_profile,
            llm=mock_llm,
            message_bus=mock_message_bus,
            config=SubAgentConfig(max_react_steps=5, max_peer_calls=3),
        )
        result = await writer.run_with_react(
            task="写一篇文章",
            context=None,
        )

        assert result["status"] == "completed"
        assert len(result["peer_calls"]) >= 1


class TestMessageBusIntegration:
    @pytest.mark.asyncio
    async def test_message_structure(self, writer_profile, mock_message_bus):
        """验证 peer_call 发送的消息结构正确"""
        mock_llm = AsyncMock()
        response = MagicMock()
        response.content = '{"thought":"需要 Review","action":"peer_call","action_input":{"target_agent":"ReviewAgent","content":"请审核"}}'
        mock_llm.ainvoke.return_value = response

        captured_msg = [None]

        async def capture_send_and_wait(msg, timeout=None):
            captured_msg[0] = msg
            reply = MagicMock()
            reply.content = {"text": "审核通过"}
            return reply

        mock_message_bus.send_and_wait = AsyncMock(side_effect=capture_send_and_wait)

        writer = SubAgent(
            profile=writer_profile,
            llm=mock_llm,
            message_bus=mock_message_bus,
            config=SubAgentConfig(max_react_steps=3),
        )
        await writer.run_with_react(
            task="审核内容",
            context=None,
        )

        msg = captured_msg[0]
        assert msg is not None
        assert msg.from_agent == "WritingAgent"
        assert msg.to_agent == "ReviewAgent"
        assert msg.msg_type == MessageType.REQUEST
