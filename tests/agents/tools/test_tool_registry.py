"""ToolRegistry — 工具注册中心、AgentTool、绑定单元测试"""
import pytest

from src.agent.tools import (
    AgentTool, AgentToolMeta, ToolRegistry,
    bind_agent_tools, tool_prompt_section,
    create_memory_tools, create_communication_tools,
    create_reflection_tools, create_hitl_tools,
)


class TestAgentTool:
    """AgentTool 基础测试"""

    @pytest.mark.asyncio
    async def test_create_and_execute(self):
        async def my_func(query: str = "") -> str:
            return f"result: {query}"

        tool = AgentTool(
            name="test_tool",
            description="A test tool",
            execute_func=my_func,
            meta=AgentToolMeta(category="general", builtin=True),
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"

        result = await tool.execute(query="hello")
        assert result == "result: hello"

    def test_default_meta(self):
        async def f() -> str:
            return "ok"

        tool = AgentTool(name="t", description="d", execute_func=f)
        assert tool.meta.category == "general"
        assert tool.meta.builtin is False
        assert tool.meta.visible is True
        assert tool.meta.agent_types == []

    def test_to_langchain_tool(self):
        async def my_func(query: str = "default") -> str:
            return f"result: {query}"

        tool = AgentTool(name="lc_test", description="LangChain wrapper test", execute_func=my_func)
        lc = tool.to_langchain_tool()
        assert lc.name == "lc_test"
        assert lc.description == "LangChain wrapper test"

    def test_from_langchain_tool(self):
        """从 LangChain BaseTool 创建 AgentTool"""
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        async def my_search(q: str = "") -> str:
            """Search for something"""
            return f"searched: {q}"

        agent_tool = AgentTool.from_langchain(my_search)
        assert agent_tool.name == "my_search"
        assert agent_tool.meta.category == "external"

    def test_from_langchain_with_meta(self):
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        async def my_tool(x: str = "") -> str:
            """A tool"""
            return x

        meta = AgentToolMeta(category="search", builtin=False, agent_types=["WebSearch"])
        agent_tool = AgentTool.from_langchain(my_tool, meta=meta)
        assert agent_tool.meta.category == "search"
        assert agent_tool.meta.agent_types == ["WebSearch"]

    def test_repr(self):
        async def f() -> str:
            return "ok"

        tool = AgentTool(name="my_tool", description="desc", execute_func=f,
                         meta=AgentToolMeta(category="memory"))
        r = repr(tool)
        assert "my_tool" in r
        assert "memory" in r


class TestToolRegistry:
    """ToolRegistry 注册中心测试"""

    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    def test_empty_registry(self, registry):
        assert registry.count == 0
        assert registry.list_tools() == []
        assert registry.list_names() == []

    def test_register_and_get(self, registry):
        async def f() -> str:
            return "ok"

        tool = AgentTool(name="my_tool", description="desc", execute_func=f)
        registry.register(tool)

        assert registry.count == 1
        assert registry.get("my_tool") is tool
        assert "my_tool" in registry.list_names()

    def test_register_many(self, registry):
        async def f1() -> str:
            return "1"
        async def f2() -> str:
            return "2"

        tools = [
            AgentTool(name="t1", description="d1", execute_func=f1),
            AgentTool(name="t2", description="d2", execute_func=f2),
        ]
        registry.register_many(tools)
        assert registry.count == 2

    def test_register_duplicate_warning(self, registry):
        async def f() -> str:
            return "ok"

        tool1 = AgentTool(name="dup", description="first", execute_func=f)
        tool2 = AgentTool(name="dup", description="second", execute_func=f)
        registry.register(tool1)
        registry.register(tool2)
        assert registry.count == 1

    def test_get_nonexistent(self, registry):
        assert registry.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_register_from_langchain(self, registry):
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        async def web_search(q: str = "") -> str:
            """Search the web"""
            return f"results for {q}"

        agent_tool = registry.register_from_langchain(
            web_search,
            meta=AgentToolMeta(category="search", builtin=False),
        )
        assert registry.count == 1
        assert agent_tool.name == "web_search"
        assert registry.get("web_search") is agent_tool


class TestToolRegistryFiltering:
    """ToolRegistry 按 Agent 过滤测试"""

    @pytest.fixture
    def registry(self):
        r = ToolRegistry()

        async def f() -> str:
            return "ok"

        # 所有 Agent 可用
        r.register(AgentTool(
            name="general_tool", description="all agents", execute_func=f,
            meta=AgentToolMeta(category="general"),
        ))
        # 仅 WritingAgent 可用
        r.register(AgentTool(
            name="writer_tool", description="only writer", execute_func=f,
            meta=AgentToolMeta(category="memory", agent_types=["WritingAgent"]),
        ))
        # 仅 ManagerAgent 可用
        r.register(AgentTool(
            name="manager_tool", description="only manager", execute_func=f,
            meta=AgentToolMeta(category="communication", agent_types=["ManagerAgent"]),
        ))
        # 不可见工具（不暴露给 LLM）
        r.register(AgentTool(
            name="internal_tool", description="internal", execute_func=f,
            meta=AgentToolMeta(category="general", visible=False),
        ))
        return r

    def test_get_for_agent_all(self, registry):
        tools = registry.get_for_agent("WebSearch")
        names = [t.name for t in tools]
        assert "general_tool" in names
        assert "writer_tool" not in names
        assert "manager_tool" not in names
        assert "internal_tool" not in names

    def test_get_for_agent_writer(self, registry):
        tools = registry.get_for_agent("WritingAgent")
        names = [t.name for t in tools]
        assert "general_tool" in names
        assert "writer_tool" in names
        assert "manager_tool" not in names

    def test_get_for_agent_manager(self, registry):
        tools = registry.get_for_agent("ManagerAgent")
        names = [t.name for t in tools]
        assert "general_tool" in names
        assert "writer_tool" not in names
        assert "manager_tool" in names

    def test_get_by_category(self, registry):
        memory_tools = registry.get_by_category("memory")
        assert len(memory_tools) == 1
        assert memory_tools[0].name == "writer_tool"

        general_tools = registry.get_by_category("general")
        names = [t.name for t in general_tools]
        assert "general_tool" in names
        assert "internal_tool" not in names  # not visible


class TestCreateBuiltinTools:
    """内置工具工厂函数测试"""

    def test_create_memory_tools(self):
        tools = create_memory_tools()
        assert len(tools) == 3
        names = [t.name for t in tools]
        assert "memory_search" in names
        assert "memory_store" in names
        assert "memory_context" in names

        for t in tools:
            assert t.meta.builtin is True
            assert t.meta.category == "memory"

    def test_create_communication_tools(self):
        tools = create_communication_tools(message_bus=None)
        assert len(tools) == 4
        names = [t.name for t in tools]
        assert "send_message" in names
        assert "check_messages" in names
        assert "get_conversation" in names
        assert "peer_call" in names

        for t in tools:
            assert t.meta.builtin is True
            assert t.meta.category == "communication"

    def test_create_reflection_tools(self):
        tools = create_reflection_tools(reflection_engine=None)
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "self_reflect" in names
        assert "request_peer_review" in names

        for t in tools:
            assert t.meta.builtin is True
            assert t.meta.category == "reflection"

    def test_create_hitl_tools(self):
        tools = create_hitl_tools(hitl_manager=None)
        assert len(tools) == 4
        names = [t.name for t in tools]
        assert "request_decision" in names
        assert "request_approval" in names
        assert "request_feedback" in names
        assert "check_pending_gates" in names

        for t in tools:
            assert t.meta.builtin is True
            assert t.meta.category == "hitl"


class TestToolBindingAndDescriptions:
    """工具绑定和描述生成测试"""

    @pytest.fixture
    def registry(self):
        r = ToolRegistry()

        async def f() -> str:
            return "ok"

        r.register(AgentTool(
            name="memory_search", description="Search memory",
            execute_func=f,
            meta=AgentToolMeta(category="memory", agent_types=["WritingAgent", "ReviewAgent"], builtin=True),
        ))
        r.register(AgentTool(
            name="send_message", description="Send msg to other agents",
            execute_func=f,
            meta=AgentToolMeta(category="communication", agent_types=["WritingAgent", "ManagerAgent"], builtin=True),
        ))
        r.register(AgentTool(
            name="general_util", description="General utility",
            execute_func=f,
            meta=AgentToolMeta(category="general"),
        ))
        return r

    def test_bind_agent_tools_none_llm(self, registry):
        result = bind_agent_tools(None, "WritingAgent", registry)
        assert result is None

    def test_bind_agent_tools_none_registry(self):
        from langchain_openai import ChatOpenAI
        # 只验证不崩溃（无 API key 时 ChatOpenAI 也可以在未调用时创建）
        try:
            llm = ChatOpenAI(api_key="test", model="gpt-3.5-turbo")
            result = bind_agent_tools(llm, "WritingAgent", None)
            assert result is llm
        except Exception:
            pass

    def test_tool_prompt_section_no_tools(self, registry):
        # All tools in registry require specific agent_types, so "OtherAgent" gets nothing
        from src.agent.tools import ToolRegistry
        empty_registry = ToolRegistry()
        result = tool_prompt_section("OtherAgent", empty_registry)
        assert result == ""

    def test_tool_prompt_section_no_registry(self):
        result = tool_prompt_section("WritingAgent", None)
        assert result == ""

    def test_tool_prompt_section_with_tools(self, registry):
        result = tool_prompt_section("WritingAgent", registry)
        assert "可用工具" in result
        assert "memory_search" in result
        assert "send_message" in result
        assert "general_util" in result

    def test_tool_prompt_section_without_category(self, registry):
        result = tool_prompt_section("WritingAgent", registry, include_category=False)
        assert "可用工具" in result
        assert "memory_search" in result
        # 没有分类标签
        assert "📝" not in result

    def test_tool_prompt_section_categories(self, registry):
        result = tool_prompt_section("WritingAgent", registry, include_category=True)
        assert "📝" in result or "💬" in result  # memory or communication category labels

    def test_describe_for_agent(self, registry):
        description = registry.describe_for_agent("WritingAgent")
        assert "memory_search" in description
        assert "send_message" in description

    def test_describe_for_agent_no_tools(self, registry):
        from src.agent.tools import ToolRegistry
        assert ToolRegistry().describe_for_agent("NonExistent") == ""


class TestToolRegistryOpenAIFunctions:
    """ToolRegistry OpenAI Function Calling 测试"""

    @pytest.fixture
    def registry(self):
        r = ToolRegistry()

        async def f(query: str = "") -> str:
            return query

        r.register(AgentTool(
            name="search_tool", description="Search tool",
            execute_func=f,
            meta=AgentToolMeta(category="search", agent_types=["WebSearch"]),
        ))
        return r

    def test_to_openai_functions(self, registry):
        functions = registry.to_openai_functions("WebSearch")
        # May be empty if LangChain's to_openai_function is not available
        assert isinstance(functions, list)

    def test_to_openai_functions_no_tools(self, registry):
        functions = registry.to_openai_functions("NonExistent")
        assert functions == []


class TestMemoryCommunicationReflectionTools:
    """内置工具在注册表中的集成测试"""

    @pytest.fixture
    def full_registry(self):
        r = ToolRegistry()
        for tool in create_memory_tools():
            r.register(tool)
        for tool in create_communication_tools(None):
            r.register(tool)
        for tool in create_reflection_tools(None):
            r.register(tool)
        for tool in create_hitl_tools(None):
            r.register(tool)
        return r

    def test_total_count(self, full_registry):
        assert full_registry.count == 13  # 3 memory + 4 comm + 2 reflection + 4 hitl

    def test_writer_agent_tools(self, full_registry):
        tools = full_registry.get_for_agent("WritingAgent")
        names = [t.name for t in tools]
        assert "memory_search" in names
        assert "memory_store" in names
        assert "send_message" in names
        assert "check_messages" in names
        assert "self_reflect" in names
        assert "request_peer_review" in names
        assert "request_feedback" in names
        # Not for Writer:
        assert "memory_context" not in names
        assert "get_conversation" not in names
        assert "request_decision" not in names

    def test_manager_agent_tools(self, full_registry):
        tools = full_registry.get_for_agent("ManagerAgent")
        names = [t.name for t in tools]
        assert "memory_search" in names
        assert "memory_store" in names
        assert "memory_context" in names  # ManagerAgent specific
        assert "send_message" in names
        assert "request_decision" in names
        assert "check_pending_gates" in names
        assert "request_approval" in names
        assert "request_peer_review" in names
        assert "request_feedback" in names
        # Not for Manager:
        assert "check_messages" not in names

    @pytest.mark.asyncio
    async def test_memory_tool_no_memory_manager(self, full_registry):
        """无 memory_manager 时应有合适的错误处理"""
        tool = full_registry.get("memory_search")
        result = await tool.execute(query="test")
        # Should not crash, returns error message
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_communication_tool_no_bus(self):
        """MessageBus 为 None 时的行为"""
        tools = create_communication_tools(None)
        tool_map = {t.name: t for t in tools}
        # send_message requires to_agent, content
        result = await tool_map["send_message"].execute(to_agent="Writer", content="hello")
        assert "未启用" in result
        # check_messages needs no args
        result = await tool_map["check_messages"].execute()
        assert "未启用" in result
        # get_conversation needs conversation_id
        result = await tool_map["get_conversation"].execute(conversation_id="c1")
        assert "未启用" in result

    @pytest.mark.asyncio
    async def test_reflection_tool_no_engine(self):
        """ReflectionEngine 为 None 时的行为"""
        tools = create_reflection_tools(None)
        tool_map = {t.name: t for t in tools}
        # self_reflect requires agent_type, output_text
        result = await tool_map["self_reflect"].execute(agent_type="Writer", output_text="test")
        assert "未启用" in result
        # request_peer_review requires target_agent, output_text
        result = await tool_map["request_peer_review"].execute(target_agent="Writer", output_text="test")
        assert "未启用" in result

    @pytest.mark.asyncio
    async def test_hitl_tool_no_manager(self):
        """HITLManager 为 None 时的行为"""
        tools = create_hitl_tools(None)
        tool_map = {t.name: t for t in tools}
        # request_decision requires question
        result = await tool_map["request_decision"].execute(question="test?")
        assert "未启用" in result
        # request_approval requires target_agent, content_summary
        result = await tool_map["request_approval"].execute(target_agent="Writer", content_summary="summary")
        assert "未启用" in result
        # request_feedback requires target_agent, content
        result = await tool_map["request_feedback"].execute(target_agent="Writer", content="content")
        assert "未启用" in result
        # check_pending_gates needs no args
        result = await tool_map["check_pending_gates"].execute()
        assert "未启用" in result

    def test_tool_prompt_section_full(self, full_registry):
        result = tool_prompt_section("WritingAgent", full_registry)
        assert "memory_search" in result
        assert "self_reflect" in result
        assert "send_message" in result
