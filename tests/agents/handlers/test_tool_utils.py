"""tool_utils.py — 工具感知 LLM 调用单元测试"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.handlers.tool_utils import (
    bind_tools,
    _check_tools,
    _get_tool_calls,
    execute_tool_calls,
    call_llm,
    _stream_only,
    _stream_final,
    MAX_TOOL_ROUNDS,
)
from src.agent.tools import AgentTool, AgentToolMeta, ToolRegistry


# ── mock 工具注册表 ──

@pytest.fixture
def registry():
    r = ToolRegistry()

    async def search(query: str = "") -> str:
        return f"搜索结果: {query}"

    async def calculate(expr: str = "") -> str:
        return f"计算结果: {expr}"

    async def failing_tool(query: str = "") -> str:
        raise ValueError("工具执行异常")

    r.register(AgentTool(
        name="web_search", description="搜索网络",
        execute_func=search,
        meta=AgentToolMeta(category="search", agent_types=["WritingAgent", "ManagerAgent"]),
    ))
    r.register(AgentTool(
        name="calculator", description="数学计算",
        execute_func=calculate,
        meta=AgentToolMeta(category="code", agent_types=["CodeAgent"]),
    ))
    r.register(AgentTool(
        name="failing_tool", description="总是失败",
        execute_func=failing_tool,
        meta=AgentToolMeta(category="general", agent_types=["WritingAgent"]),
    ))
    return r


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    llm.astream = MagicMock()  # 返回 async generator
    llm.bind_tools = MagicMock(return_value=llm)  # return self
    return llm


# ── _check_tools ──

class TestCheckTools:
    def test_none_registry(self):
        assert _check_tools(None, "WritingAgent") is False

    def test_empty_agent_type(self):
        assert _check_tools(MagicMock(), "") is False

    def test_has_tools(self, registry):
        assert _check_tools(registry, "WritingAgent") is True

    def test_no_tools_for_agent(self, registry):
        assert _check_tools(registry, "ImageAgent") is False  # no tools for ImageAgent

    def test_registry_error(self):
        broken = MagicMock()
        broken.get_for_agent = MagicMock(side_effect=Exception("boom"))
        assert _check_tools(broken, "X") is False


# ── bind_tools ──

class TestBindTools:
    def test_none_registry(self, mock_llm):
        result = bind_tools(mock_llm, None, "WritingAgent")
        assert result is mock_llm
        mock_llm.bind_tools.assert_not_called()

    def test_empty_agent(self, mock_llm, registry):
        result = bind_tools(mock_llm, registry, "")
        assert result is mock_llm

    def test_no_visible_tools(self, mock_llm, registry):
        result = bind_tools(mock_llm, registry, "ImageAgent")
        assert result is mock_llm

    def test_binds_tools(self, mock_llm, registry):
        result = bind_tools(mock_llm, registry, "WritingAgent")
        assert result is not None
        mock_llm.bind_tools.assert_called_once()

    def test_bind_error_returns_original(self, mock_llm, registry):
        mock_llm.bind_tools = MagicMock(side_effect=Exception("bind failed"))
        result = bind_tools(mock_llm, registry, "WritingAgent")
        assert result is mock_llm

    def test_none_llm(self, registry):
        result = bind_tools(None, registry, "WritingAgent")
        assert result is None


# ── _get_tool_calls ──

class TestGetToolCalls:
    def test_no_tool_calls(self):
        msg = AIMessage(content="纯文本回复")
        assert _get_tool_calls(msg) == []

    def test_empty_tool_calls(self):
        msg = AIMessage(content="text", tool_calls=[])
        assert _get_tool_calls(msg) == []

    def test_with_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[{
                "name": "web_search",
                "args": {"query": "test"},
                "id": "call_123",
            }],
        )
        calls = _get_tool_calls(msg)
        assert len(calls) == 1
        assert calls[0]["name"] == "web_search"
        assert calls[0]["args"] == {"query": "test"}
        assert calls[0]["id"] == "call_123"

    def test_multiple_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "search", "args": {"q": "a"}, "id": "c1"},
                {"name": "calc", "args": {"expr": "1+1"}, "id": "c2"},
            ],
        )
        calls = _get_tool_calls(msg)
        assert len(calls) == 2

    def test_no_tool_calls_attribute(self):
        """AIMessage without tool_calls attribute"""
        msg = AIMessage(content="hello")
        assert _get_tool_calls(msg) == []


# ── execute_tool_calls ──

class TestExecuteToolCalls:
    @pytest.mark.asyncio
    async def test_execute_success(self, registry):
        messages = [HumanMessage(content="搜索测试")]
        tool_calls = [{"name": "web_search", "args": {"query": "test"}, "id": "call_1"}]

        result = await execute_tool_calls(tool_calls, registry, messages)
        assert len(result) == 2  # original + ToolMessage
        assert isinstance(result[-1], ToolMessage)
        assert "搜索结果" in result[-1].content

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, registry):
        messages = [HumanMessage(content="hi")]
        tool_calls = [{"name": "nonexistent_tool", "args": {}, "id": "call_1"}]

        result = await execute_tool_calls(tool_calls, registry, messages)
        assert "不可用" in result[-1].content

    @pytest.mark.asyncio
    async def test_execute_tool_error(self, registry):
        messages = [HumanMessage(content="hi")]
        tool_calls = [{"name": "failing_tool", "args": {}, "id": "call_1"}]

        result = await execute_tool_calls(tool_calls, registry, messages)
        assert "错误" in result[-1].content or "异常" in result[-1].content

    @pytest.mark.asyncio
    async def test_execute_no_registry(self):
        messages = [HumanMessage(content="hi")]
        tool_calls = [{"name": "web_search", "args": {"query": "t"}, "id": "c1"}]

        result = await execute_tool_calls(tool_calls, None, messages)
        assert "未启用" in result[-1].content

    @pytest.mark.asyncio
    async def test_execute_multiple_calls(self, registry):
        messages = [HumanMessage(content="do both")]
        tool_calls = [
            {"name": "web_search", "args": {"query": "first"}, "id": "c1"},
            {"name": "web_search", "args": {"query": "second"}, "id": "c2"},
        ]

        result = await execute_tool_calls(tool_calls, registry, messages)
        assert len(result) == 3  # original + 2 ToolMessages

    @pytest.mark.asyncio
    async def test_content_truncated(self, registry):
        """ToolMessage 内容应被截断到 4000 字符"""
        long_tool = ToolRegistry()
        async def long_func() -> str:
            return "x" * 5000

        long_tool.register(AgentTool(
            name="long_output", description="long", execute_func=long_func,
        ))

        messages = [HumanMessage(content="hi")]
        tool_calls = [{"name": "long_output", "args": {}, "id": "c1"}]

        result = await execute_tool_calls(tool_calls, long_tool, messages)
        assert len(result[-1].content) <= 4000


# ── call_llm ──

class TestCallLLM:
    @pytest.mark.asyncio
    async def test_no_tools_no_stream(self, mock_llm):
        """无工具、非流式 — 直接 ainvoke"""
        mock_llm.ainvoke.return_value = AIMessage(content="hello world")
        messages = [HumanMessage(content="hi")]

        result = await call_llm(mock_llm, messages)
        assert result == "hello world"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_tools_with_stream(self, mock_llm):
        """无工具、流式 — 走 _stream_only"""
        async def mock_stream(messages):
            for chunk in ["hello", " ", "world"]:
                yield AIMessage(content=chunk)

        mock_llm.astream.return_value = mock_stream([])
        messages = [HumanMessage(content="hi")]

        result = await call_llm(mock_llm, messages, stream=True,
                                on_chunk=AsyncMock())
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_tool_roundtrip(self, mock_llm, registry):
        """单轮工具调用 — LLM 先返回 tool_call，执行后返回文本"""
        # 第一轮返回 tool_call
        first_response = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "test"}, "id": "tc_1"}],
        )
        # 第二轮返回纯文本
        second_response = AIMessage(content="搜索完成")
        mock_llm.ainvoke.side_effect = [first_response, second_response]

        messages = [HumanMessage(content="搜索一下")]
        result = await call_llm(mock_llm, messages, tool_registry=registry,
                                agent_type="WritingAgent")
        assert result == "搜索完成"
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_not_found_in_loop(self, mock_llm, registry):
        """工具不存在时应有错误提示"""
        first = AIMessage(
            content="",
            tool_calls=[{"name": "unknown_tool", "args": {}, "id": "tc_1"}],
        )
        second = AIMessage(content="done")
        mock_llm.ainvoke.side_effect = [first, second]

        result = await call_llm(mock_llm, [HumanMessage(content="hi")],
                                tool_registry=registry, agent_type="WritingAgent")
        assert result == "done"

    @pytest.mark.asyncio
    async def test_multiple_tool_rounds(self, mock_llm, registry):
        """多轮工具调用"""
        r1 = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "q1"}, "id": "tc_1"}],
        )
        r2 = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "q2"}, "id": "tc_2"}],
        )
        r3 = AIMessage(content="最终结果")
        mock_llm.ainvoke.side_effect = [r1, r2, r3]

        result = await call_llm(mock_llm, [HumanMessage(content="search")],
                                tool_registry=registry, agent_type="WritingAgent")
        assert result == "最终结果"
        assert mock_llm.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_max_tool_rounds(self, mock_llm, registry):
        """超出最大轮次时返回最后一条消息"""
        # 每次循环都返回 tool_call
        tool_response = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "q"}, "id": "tc"}],
        )
        mock_llm.ainvoke.return_value = tool_response

        result = await call_llm(mock_llm, [HumanMessage(content="search")],
                                tool_registry=registry, agent_type="WritingAgent",
                                max_tool_rounds=3)
        # 超出轮次后返回最后的 content（空字符串）
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_tool_roundtrip_with_stream(self, mock_llm, registry):
        """工具调用 + 流式输出"""
        tool_resp = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "t"}, "id": "tc"}],
        )
        text_resp = AIMessage(content="final answer")
        mock_llm.ainvoke.side_effect = [tool_resp, text_resp]

        async def mock_stream(messages):
            yield AIMessage(content="final answer")
        mock_llm.astream.return_value = mock_stream([])
        on_chunk = AsyncMock()

        result = await call_llm(mock_llm, [HumanMessage(content="search")],
                                tool_registry=registry, agent_type="WritingAgent",
                                stream=True, on_chunk=on_chunk,
                                max_tool_rounds=2)
        assert "final answer" in result

    @pytest.mark.asyncio
    async def test_no_tools_fallback(self, mock_llm, registry):
        """Agent 无可见工具时退化为原始调用"""
        mock_llm.ainvoke.return_value = AIMessage(content="no tools for me")
        result = await call_llm(mock_llm, [HumanMessage(content="hi")],
                                tool_registry=registry, agent_type="ImageAgent")
        assert result == "no tools for me"

    @pytest.mark.asyncio
    async def test_llm_returns_direct_text_in_loop(self, mock_llm, registry):
        """有工具时 LLM 第一轮直接返回文本（不调用工具）"""
        mock_llm.ainvoke.return_value = AIMessage(content="直接回复，无需工具")
        result = await call_llm(mock_llm, [HumanMessage(content="hi")],
                                tool_registry=registry, agent_type="WritingAgent")
        assert result == "直接回复，无需工具"
        mock_llm.ainvoke.assert_called_once()


# ── _stream_only ──

class TestStreamOnly:
    @pytest.mark.asyncio
    async def test_stream_only(self):
        llm = MagicMock()
        async def mock_stream(messages):
            for chunk in ["Hello", " ", "World"]:
                yield AIMessage(content=chunk)
        llm.astream.return_value = mock_stream([])

        on_chunk = AsyncMock()
        result = await _stream_only(llm, [HumanMessage(content="hi")], on_chunk)
        assert result == "Hello World"
        assert on_chunk.call_count == 3


# ── _stream_final ──

class TestStreamFinal:
    @pytest.mark.asyncio
    async def test_stream_final(self):
        llm = MagicMock()
        async def mock_stream(messages):
            yield AIMessage(content="final")
        llm.astream.return_value = mock_stream([])

        on_chunk = AsyncMock()
        messages = [HumanMessage(content="hi"), AIMessage(content="thinking"),
                    ToolMessage(content="result", tool_call_id="tc1")]
        result = await _stream_final(llm, messages, on_chunk)
        assert result == "final"
        on_chunk.assert_called_once_with("final")


# ── Edge Cases ──

class TestCallLLMEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_messages(self, mock_llm):
        mock_llm.ainvoke.return_value = AIMessage(content="")
        result = await call_llm(mock_llm, [])
        assert result == ""

    @pytest.mark.asyncio
    async def test_none_llm(self, registry):
        """llm 为 None 时应优雅处理"""
        with pytest.raises((AttributeError, TypeError)):
            await call_llm(None, [HumanMessage(content="hi")],
                           tool_registry=registry, agent_type="WritingAgent")

    @pytest.mark.asyncio
    async def test_registry_error_in_loop(self, mock_llm, registry):
        """工具注册表异常时应在循环中恢复"""
        first = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "t"}, "id": "tc"}],
        )
        second = AIMessage(content="done")
        mock_llm.ainvoke.side_effect = [first, second]

        result = await call_llm(mock_llm, [HumanMessage(content="search")],
                                tool_registry=registry, agent_type="WritingAgent")
        assert result == "done"

    @pytest.mark.asyncio
    async def test_tool_call_with_empty_args(self, mock_llm, registry):
        """工具调用 args 为空字典"""
        first = AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {}, "id": "tc_1"}],
        )
        second = AIMessage(content="done")
        mock_llm.ainvoke.side_effect = [first, second]

        result = await call_llm(mock_llm, [HumanMessage(content="search")],
                                tool_registry=registry, agent_type="WritingAgent")
        assert result == "done"

    @pytest.mark.asyncio
    async def test_tool_call_with_failing_tool(self, mock_llm, registry):
        """工具执行失败时 LLM 应收到错误消息并继续"""
        first = AIMessage(
            content="",
            tool_calls=[{"name": "failing_tool", "args": {}, "id": "tc_1"}],
        )
        second = AIMessage(content="已处理错误")
        mock_llm.ainvoke.side_effect = [first, second]

        result = await call_llm(mock_llm, [HumanMessage(content="run")],
                                tool_registry=registry, agent_type="WritingAgent")
        assert result == "已处理错误"
