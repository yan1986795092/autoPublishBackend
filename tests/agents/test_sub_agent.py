"""SubAgent ReAct 循环测试"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agent.react.sub_agent import SubAgent
from src.agent.agents.profile import AgentProfile
from src.agent.schemas import SubAgentConfig


@pytest.fixture
def writing_profile():
    return AgentProfile(
        name="WritingAgent",
        role="内容创作者",
        description="多风格内容写作",
        persona="你是一位经验丰富的写作者。",
        expertise=["博客", "社交媒体"],
        communication_style="自然流畅",
        token_budget=4000,
        memory_namespace="content",
        memory_enabled=True,
        remember_enabled=True,
        tools_enabled=True,
        handler_key="writing",
    )


@pytest.fixture
def subagent_config():
    return SubAgentConfig(
        max_react_steps=5,
        max_peer_calls=3,
        react_timeout=30.0,
        enable_peer_collaboration=True,
        peer_call_timeout=10.0,
    )


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    response = MagicMock()
    response.content = '{"thought":"任务简单，直接完成","action":"finish","action_input":{}}'
    llm.ainvoke.return_value = response
    return llm


@pytest.fixture
def mock_tool_registry():
    reg = MagicMock()
    tool = MagicMock()
    tool.name = "web_search"
    tool.description = "搜索网络"
    tool.execute = AsyncMock(return_value="搜索结果：赛博朋克2077由CDPR开发")
    reg.get.return_value = tool
    reg.get_for_agent.return_value = [tool]
    return reg


@pytest.fixture
def mock_message_bus():
    return MagicMock()


class TestSubAgentInit:
    def test_init_with_profile(self, writing_profile, subagent_config):
        agent = SubAgent(
            profile=writing_profile,
            llm=None,
            config=subagent_config,
        )
        assert agent.name == "WritingAgent"
        assert agent.role == "内容创作者"
        assert agent._config.max_react_steps == 5

    def test_init_without_config(self, writing_profile):
        agent = SubAgent(profile=writing_profile)
        assert agent._config.max_react_steps == 8


class TestSubAgentReActLoop:
    @pytest.mark.asyncio
    async def test_react_finish_without_llm(self, writing_profile):
        """无 LLM 时，规则引擎应能完成简单任务"""
        agent = SubAgent(
            profile=writing_profile,
            llm=None,
            config=SubAgentConfig(max_react_steps=3),
        )
        result = await agent.run_with_react(
            task="写一篇关于AI的短文",
            context=None,
        )
        assert "output" in result
        assert "react_steps" in result
        assert result["status"] in ("completed", "partial")

    @pytest.mark.asyncio
    async def test_react_with_mock_llm_finish(self, writing_profile, mock_llm, subagent_config):
        """LLM 返回 finish 时，循环应立即结束"""
        agent = SubAgent(
            profile=writing_profile,
            llm=mock_llm,
            config=subagent_config,
        )
        result = await agent.run_with_react(
            task="介绍量子计算",
            context=None,
        )
        assert result["status"] == "completed"
        assert len(result["react_steps"]) <= subagent_config.max_react_steps

    @pytest.mark.asyncio
    async def test_react_with_peer_call(self, writing_profile, subagent_config, mock_message_bus):
        """LLM 返回 peer_call 时，应调用 MessageBus"""
        mock_llm = AsyncMock()
        response = MagicMock()
        response.content = '{"thought":"需要审核","action":"peer_call","action_input":{"target_agent":"ReviewAgent","content":"请审核这段文字"}}'
        mock_llm.ainvoke.return_value = response

        mock_message_bus.send_and_wait = AsyncMock(
            return_value=MagicMock(
                content={"text": "审核通过，文章质量良好"},
            )
        )

        agent = SubAgent(
            profile=writing_profile,
            llm=mock_llm,
            message_bus=mock_message_bus,
            config=subagent_config,
        )
        result = await agent.run_with_react(
            task="写一段赛博朋克风格的开场白",
            context=None,
        )
        assert "output" in result
        assert "react_steps" in result
        mock_message_bus.send_and_wait.assert_called()

    @pytest.mark.asyncio
    async def test_react_max_steps_limit(self, writing_profile):
        """达到最大步数时应该停止"""
        agent = SubAgent(
            profile=writing_profile,
            llm=None,
            config=SubAgentConfig(max_react_steps=2),
        )
        result = await agent.run_with_react(
            task="写一个故事",
            context=None,
        )
        assert len(result["react_steps"]) <= 2

    @pytest.mark.asyncio
    async def test_react_with_dependency_context(self, writing_profile, mock_llm, subagent_config):
        """context 包含 dependencies 时应正确序列化"""
        agent = SubAgent(
            profile=writing_profile,
            llm=mock_llm,
            config=subagent_config,
        )
        result = await agent.run_with_react(
            task="基于调研结果写文章",
            context={
                "dependencies": {
                    "dep-1": {"content": "调研发现AI在医疗领域应用广泛", "agent": "ResearcherAgent"}
                },
                "session_id": "test-session",
            },
        )
        assert "output" in result


class TestPeerCallLimit:
    @pytest.mark.asyncio
    async def test_peer_call_respects_limit(self, writing_profile):
        """peer_call 达到上限后不再调用 MessageBus"""
        mock_llm = AsyncMock()
        response = MagicMock()
        response.content = '{"thought":"需要编辑","action":"peer_call","action_input":{"target_agent":"EditorAgent","content":"请润色"}}'
        mock_llm.ainvoke.return_value = response

        mock_bus = MagicMock()
        mock_bus.send_and_wait = AsyncMock(return_value=MagicMock(content={"text": "已润色"}))

        agent = SubAgent(
            profile=writing_profile,
            llm=mock_llm,
            message_bus=mock_bus,
            config=SubAgentConfig(max_react_steps=10, max_peer_calls=2),
        )
        result = await agent.run_with_react(
            task="写一段话",
            context=None,
        )
        successful_calls = [c for c in result.get("peer_calls", []) if c.get("success")]
        blocked_calls = [c for c in result.get("peer_calls", []) if not c.get("success")]
        assert len(successful_calls) == 2, f"期望 2 次成功调用，实际 {len(successful_calls)}"
        assert all("最大 peer_call" in c.get("error", "") for c in blocked_calls), "被阻止的调用应有明确的错误信息"

    @pytest.mark.asyncio
    async def test_peer_call_disabled(self, writing_profile):
        """对等协作未启用时，peer_call 返回错误"""
        mock_llm = AsyncMock()
        response = MagicMock()
        response.content = '{"thought":"需要编辑","action":"peer_call","action_input":{"target_agent":"EditorAgent","content":"润色"}}'
        mock_llm.ainvoke.return_value = response

        agent = SubAgent(
            profile=writing_profile,
            llm=mock_llm,
            message_bus=None,
            config=SubAgentConfig(enable_peer_collaboration=False),
        )
        result = await agent.run_with_react(
            task="写一段话",
            context=None,
        )
        peer_calls = result.get("peer_calls", [])
        assert len(peer_calls) > 0
        assert not peer_calls[0]["success"]
        assert peer_calls[0]["error"] is not None


class TestRuleBasedDecisions:
    @pytest.mark.asyncio
    async def test_rule_engine_peer_signals(self, writing_profile):
        """规则引擎应识别 peer 协作信号"""
        mock_llm = AsyncMock()
        response = MagicMock()
        response.content = "我会继续写这段文字，需要事实核查。"
        mock_llm.ainvoke.return_value = response

        mock_bus = MagicMock()
        mock_bus.send_and_wait = AsyncMock(return_value=MagicMock(content={"text": "核实完成"}))

        agent = SubAgent(
            profile=writing_profile,
            llm=mock_llm,
            message_bus=mock_bus,
            config=SubAgentConfig(max_react_steps=5, max_peer_calls=3),
        )
        result = await agent.run_with_react(
            task="写一段科技文章",
            context=None,
        )
        assert "output" in result

    def test_guess_peer_target(self, writing_profile):
        """_guess_peer_target 应根据文本内容正确猜测目标 Agent"""
        agent = SubAgent(profile=writing_profile)
        assert agent._guess_peer_target("这段话需要审核") == "ReviewAgent"
        assert agent._guess_peer_target("需要润色改写") == "EditorAgent"
        assert agent._guess_peer_target("事实核查一下") == "ResearcherAgent"
        assert agent._guess_peer_target("关键词seo排名") == "SEOAgent"
        assert agent._guess_peer_target("无关内容") == "ReviewAgent"
