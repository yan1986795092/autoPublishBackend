"""AgentRegistry — 注册中心、Profile、BaseAgent 单元测试"""
import pytest
from dataclasses import asdict

from src.agent.agents import AgentProfile, PREDEFINED_PROFILES, AgentRegistry, BaseAgent


class TestAgentProfile:
    """AgentProfile 数据类测试"""

    def test_default_values(self):
        p = AgentProfile(name="TestAgent", role="测试员")
        assert p.name == "TestAgent"
        assert p.role == "测试员"
        assert p.description == ""
        assert p.persona == ""
        assert p.expertise == []
        assert p.token_budget == 2000
        assert p.tools_enabled is False
        assert p.cacheable is False
        assert p.memory_enabled is False
        assert p.remember_enabled is False

    def test_full_profile(self):
        p = AgentProfile(
            name="WritingAgent",
            role="内容创作者",
            description="多风格内容写作",
            persona="你是一位经验丰富的写作者",
            expertise=["博客", "技术写作"],
            communication_style="自然流畅",
            token_budget=4000,
            memory_namespace="content",
            tools_enabled=True,
            cacheable=False,
            memory_enabled=True,
            remember_enabled=True,
        )
        assert p.token_budget == 4000
        assert p.tools_enabled is True

    def test_to_dict(self):
        p = AgentProfile(name="Test", role="测试", description="test")
        d = p.to_dict()
        assert d["name"] == "Test"
        assert d["role"] == "测试"
        assert d["description"] == "test"

    def test_from_dict(self):
        data = {"name": "FromDict", "role": "导入", "token_budget": 3000}
        p = AgentProfile.from_dict(data)
        assert p.name == "FromDict"
        assert p.role == "导入"
        assert p.token_budget == 3000
        assert p.description == ""  # not in data, uses default

    def test_from_dict_ignores_unknown_fields(self):
        data = {"name": "X", "role": "Y", "unknown_field": "should_ignore"}
        p = AgentProfile.from_dict(data)
        assert p.name == "X"
        assert not hasattr(p, "unknown_field")


class TestPredefinedProfiles:
    """预定义 Profile 完整性测试"""

    def test_has_all_expected_types(self):
        expected = {
            "ManagerAgent", "WritingAgent", "ReviewAgent", "SEOAgent",
            "DiagramAgent", "CodeAgent", "WebSearch", "HotTopicAgent",
            "ImageAgent", "PlatformAdapter", "TranslationAgent", "URLCrawler",
        }
        assert set(PREDEFINED_PROFILES.keys()) == expected

    def test_all_profiles_have_name_and_role(self):
        for name, p in PREDEFINED_PROFILES.items():
            assert p.name == name, f"{name}: name mismatch"
            assert p.role, f"{name}: role is empty"

    def test_token_budgets_match(self):
        expected = {
            "ManagerAgent": 4000, "WritingAgent": 4000, "ReviewAgent": 1500,
            "SEOAgent": 1000, "DiagramAgent": 1500, "CodeAgent": 2000,
            "WebSearch": 500, "HotTopicAgent": 500, "ImageAgent": 500,
            "PlatformAdapter": 1000, "TranslationAgent": 1500, "URLCrawler": 0,
        }
        for name, budget in expected.items():
            assert PREDEFINED_PROFILES[name].token_budget == budget, f"{name}: budget mismatch"

    def test_tools_enabled_agents(self):
        enabled = [n for n, p in PREDEFINED_PROFILES.items() if p.tools_enabled]
        assert "WritingAgent" in enabled
        assert "SEOAgent" in enabled
        assert "ReviewAgent" in enabled
        assert "CodeAgent" in enabled
        assert "TranslationAgent" in enabled
        assert "ManagerAgent" in enabled
        assert "URLCrawler" not in enabled

    def test_cacheable_agents(self):
        cacheable = [n for n, p in PREDEFINED_PROFILES.items() if p.cacheable]
        assert "ReviewAgent" in cacheable
        assert "SEOAgent" in cacheable
        assert "WebSearch" in cacheable
        assert "HotTopicAgent" in cacheable
        assert "PlatformAdapter" in cacheable
        assert "URLCrawler" in cacheable

    def test_memory_enabled_agents(self):
        enabled = [n for n, p in PREDEFINED_PROFILES.items() if p.memory_enabled]
        assert "WritingAgent" in enabled
        assert "ReviewAgent" in enabled
        assert "SEOAgent" in enabled
        assert "CodeAgent" in enabled
        assert "TranslationAgent" in enabled

    def test_remember_enabled_agents(self):
        enabled = [n for n, p in PREDEFINED_PROFILES.items() if p.remember_enabled]
        assert "WritingAgent" in enabled
        assert "SEOAgent" in enabled
        assert "DiagramAgent" in enabled
        assert "WebSearch" in enabled
        # 以下 Agent 未启用 remember_enabled
        assert "CodeAgent" not in enabled
        assert "ReviewAgent" not in enabled


class TestAgentRegistry:
    """AgentRegistry 测试"""

    def test_empty_registry(self):
        registry = AgentRegistry(profiles={})
        assert registry.list_types() == []
        assert registry.get("WritingAgent") is None

    def test_register_and_get(self):
        registry = AgentRegistry()
        agent = registry.get("WritingAgent")
        assert agent is not None
        assert agent.name == "WritingAgent"
        assert agent.role == "内容创作者"

    def test_get_profile(self):
        registry = AgentRegistry()
        p = registry.get_profile("ManagerAgent")
        assert p is not None
        assert p.role == "内容生产经理"

    def test_get_nonexistent(self):
        registry = AgentRegistry()
        assert registry.get("NonExistent") is None
        assert registry.get_profile("NonExistent") is None

    def test_list_types(self):
        registry = AgentRegistry()
        types = registry.list_types()
        assert "WritingAgent" in types
        assert "ManagerAgent" in types
        assert "ReviewAgent" in types

    def test_list_profiles(self):
        registry = AgentRegistry()
        profiles = registry.list_profiles()
        assert len(profiles) == len(PREDEFINED_PROFILES)

    def test_find_by_expertise(self):
        registry = AgentRegistry()
        results = registry.find_by_expertise("博客")
        assert len(results) >= 1
        assert any(p.name == "WritingAgent" for p in results)

    def test_find_by_expertise_no_match(self):
        registry = AgentRegistry()
        results = registry.find_by_expertise("nonexistent_expertise_xyz")
        assert results == []

    def test_find_by_memory_enabled(self):
        registry = AgentRegistry()
        enabled = registry.find_by_memory_enabled()
        assert "WritingAgent" in enabled
        assert "ReviewAgent" in enabled
        assert "ManagerAgent" not in enabled  # ManagerAgent has memory_enabled=False

    def test_find_cacheable(self):
        registry = AgentRegistry()
        cacheable = registry.find_cacheable()
        assert "ReviewAgent" in cacheable
        assert "SEOAgent" in cacheable

    def test_find_tool_enabled(self):
        registry = AgentRegistry()
        enabled = registry.find_tool_enabled()
        assert "WritingAgent" in enabled
        assert "CodeAgent" in enabled

    def test_register_new_agent(self):
        registry = AgentRegistry(profiles={})
        assert registry.list_types() == []

        profile = AgentProfile(name="CustomAgent", role="定制", token_budget=1000)
        agent = BaseAgent(profile=profile)
        registry.register("CustomAgent", agent)

        assert registry.get("CustomAgent") is not None
        assert "CustomAgent" in registry.list_types()

    def test_register_overwrite(self):
        registry = AgentRegistry(profiles={})
        p1 = AgentProfile(name="A1", role="第一版")
        p2 = AgentProfile(name="A1", role="第二版")
        registry.register("A1", BaseAgent(profile=p1))
        registry.register("A1", BaseAgent(profile=p2))
        assert registry.get_profile("A1").role == "第二版"

    def test_repr(self):
        registry = AgentRegistry()
        r = repr(registry)
        assert "AgentRegistry" in r
        assert "12 agents" in r


class TestBaseAgent:
    """BaseAgent 基础功能测试"""

    def test_create_with_profile(self):
        p = AgentProfile(name="TestAgent", role="测试员")
        agent = BaseAgent(profile=p)
        assert agent.name == "TestAgent"
        assert agent.role == "测试员"

    @pytest.mark.asyncio
    async def test_execute_without_handler(self):
        """没有对应 handler 的 Agent 应返回 fallback"""
        p = AgentProfile(name="NonExistentHandler", role="测试")
        agent = BaseAgent(profile=p)
        result = await agent.execute({"topic": "test"})
        assert result.get("status") in ("completed", "partial")

    @pytest.mark.asyncio
    async def test_execute_injects_profile(self):
        """execute 应在 task_input 中注入 _agent_profile"""
        p = AgentProfile(name="NonExistentHandler", role="测试员")
        agent = BaseAgent(profile=p)

        captured = {}

        async def fake_handler(task_input, llm, mcp_tools):
            captured["profile"] = task_input.get("_agent_profile")
            return {"status": "completed", "output": {"content": "ok"}}

        agent._handler_fn = fake_handler
        await agent.execute({"topic": "test"})
        assert captured["profile"] is p

    @pytest.mark.asyncio
    async def test_execute_with_memory_injects_context(self):
        """execute_with_memory 应注入 _memory_context"""
        p = AgentProfile(name="NonExistentHandler", role="测试")
        agent = BaseAgent(profile=p)

        captured = {}

        async def fake_handler(task_input, llm, mcp_tools):
            captured["memory"] = task_input.get("_memory_context")
            return {"status": "completed", "output": {"content": "ok"}}

        agent._handler_fn = fake_handler
        await agent.execute_with_memory({"topic": "test"}, memory_context="ctx data")
        assert captured["memory"] == "ctx data"
