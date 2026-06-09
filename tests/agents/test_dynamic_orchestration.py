"""Dynamic Agent Orchestration — 集成测试

覆盖范围：
- LLM-primary planning (classify_and_decompose)
- Template fallback (_build_fallback_plan)
- Plan structure validation
- Agent-to-agent peer feedback via MessageBus
- AgentRegistry capability matching
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
import json, time

from src.agent.nodes import classify_and_decompose, _build_fallback_plan, schedule_and_execute
from src.agent.orchestrator import Orchestrator
from src.agent.schemas import AgentTask, ExecutionPlan
from src.agent.agent_utils import validate_plan_structure, extract_plan_from_llm_response
from src.agent.communication import MessageBus, MessageType


# ══════════════════════════════════════════════
# _build_fallback_plan 单元测试
# ══════════════════════════════════════════════

class TestFallbackPlan:
    """验证模板兜底计划生成（与 schema 新字段兼容）"""

    def test_simple_returns_two_tasks(self):
        plan = _build_fallback_plan("写一篇关于AI的博客", "simple")
        assert len(plan["tasks"]) == 2
        assert plan["plan_method"] == "template"
        assert plan["goal"] == "写一篇关于AI的博客"

    def test_medium_returns_four_tasks(self):
        plan = _build_fallback_plan("为小红书写一篇Python教程", "medium")
        assert len(plan["tasks"]) == 4
        agent_types = {t["agent_type"] for t in plan["tasks"]}
        assert "WebSearch" in agent_types
        assert "WritingAgent" in agent_types
        assert "SEOAgent" in agent_types
        assert "ReviewAgent" in agent_types

    def test_pure_copywriting_request_should_classify_simple(self):
        orch = Orchestrator()
        assert orch._classify_complexity("为一款智能笔记应用写产品推广文案") == "simple"

    def test_complex_includes_diagram_agent(self):
        plan = _build_fallback_plan("创建微服务架构全面指南", "complex")
        agent_types = {t["agent_type"] for t in plan["tasks"]}
        assert "DiagramAgent" in agent_types
        assert "WebSearch" in agent_types
        assert "WritingAgent" in agent_types

    def test_all_tasks_have_required_fields(self):
        plan = _build_fallback_plan("测试请求", "complex")
        for task in plan["tasks"]:
            assert "id" in task
            assert "agent_type" in task
            assert "input" in task
            assert "dependencies" in task
            assert "can_parallel_with" in task
            assert "retry_on_low_quality" in task
            assert "priority" in task

    def test_writing_agent_has_retry_on_low_quality(self):
        plan = _build_fallback_plan("测试", "complex")
        for task in plan["tasks"]:
            if task["agent_type"] == "WritingAgent":
                assert task["retry_on_low_quality"] is True
            if task["agent_type"] == "WebSearch":
                assert task["retry_on_low_quality"] is False

    def test_dependency_chain_valid(self):
        plan = _build_fallback_plan("测试", "complex")
        task_ids = {t["id"] for t in plan["tasks"]}
        for task in plan["tasks"]:
            for dep in task.get("dependencies", []):
                assert isinstance(dep, int), f"Dependency should be int index, got {type(dep)}"
                assert 0 <= dep < len(plan["tasks"]), f"Dependency index {dep} out of range"


# ══════════════════════════════════════════════
# Plan validation / extraction 工具测试
# ══════════════════════════════════════════════

class TestPlanValidation:
    """测试 plan 结构验证和 LLM response 提取"""

    def test_valid_plan_passes(self):
        plan = {
            "goal": "test",
            "reasoning": "test reasoning",
            "tasks": [
                {"agent_type": "WritingAgent", "input": {"topic": "test"}, "dependencies": []},
                {"agent_type": "ReviewAgent", "input": {"topic": "test"}, "dependencies": [0]},
            ],
        }
        is_valid, msg = validate_plan_structure(plan)
        assert is_valid is True
        assert msg == ""

    def test_missing_goal_fails(self):
        plan = {
            "reasoning": "test",
            "tasks": [],
        }
        is_valid, msg = validate_plan_structure(plan)
        assert is_valid is False
        assert "goal" in msg.lower() or "missing" in msg.lower()

    def test_empty_tasks_fails(self):
        plan = {"goal": "test", "reasoning": "x", "tasks": []}
        is_valid, msg = validate_plan_structure(plan)
        assert is_valid is False
        assert "empty" in msg.lower()

    def test_unknown_agent_type_fails(self):
        plan = {
            "goal": "test", "reasoning": "x",
            "tasks": [{"agent_type": "FakeAgent", "input": {}, "dependencies": []}],
        }
        is_valid, msg = validate_plan_structure(plan)
        assert is_valid is False
        assert "FakeAgent" in msg

    def test_invalid_dependency_index_fails(self):
        plan = {
            "goal": "test", "reasoning": "x",
            "tasks": [
                {"agent_type": "WritingAgent", "input": {}, "dependencies": [99]},
            ],
        }
        is_valid, msg = validate_plan_structure(plan)
        assert is_valid is False
        assert "invalid" in msg.lower() or "dependency" in msg.lower()


class TestPlanExtraction:
    """测试从 LLM response 中提取 plan JSON"""

    def test_extract_from_code_block(self):
        response = '''Here is the plan:
```json
{
  "goal": "test goal",
  "reasoning": "test reasoning",
  "tasks": [
    {"agent_type": "WritingAgent", "input": {"topic": "AI"}, "dependencies": []}
  ]
}
```
'''
        plan = extract_plan_from_llm_response(response)
        assert plan["goal"] == "test goal"
        assert len(plan["tasks"]) == 1

    def test_extract_from_plain_json(self):
        response = '{"goal":"test","reasoning":"x","tasks":[{"agent_type":"SEOAgent","input":{},"dependencies":[]}]}'
        plan = extract_plan_from_llm_response(response)
        assert plan["goal"] == "test"
        assert plan["tasks"][0]["agent_type"] == "SEOAgent"

    def test_extract_from_raw_json_object(self):
        response = '''Some text before {"goal":"raw","reasoning":"y","tasks":[{"agent_type":"WebSearch","input":{"query":"test"},"dependencies":[]}]} some text after'''
        plan = extract_plan_from_llm_response(response)
        assert plan["goal"] == "raw"

    def test_extract_invalid_raises(self):
        with pytest.raises(ValueError):
            extract_plan_from_llm_response("This is not JSON at all")


# ══════════════════════════════════════════════
# classify_and_decompose — LLM primary path 测试
# ══════════════════════════════════════════════

class TestClassifyDecomposeLLMPrimary:
    """验证 LLM 是主路径（不只是 medium/complex）"""

    @pytest.fixture
    def base_state(self):
        return {
            "user_request": "写一篇博客文章",
            "context": {},
            "plan": None,
            "plan_reasoning": "",
            "task_queue": [],
            "completed_task_ids": [],
            "failed_task_ids": [],
            "task_results": {},
            "ready_tasks": [],
            "concurrency_limit": 3,
            "thinking_steps": [],
            "trace_task_id": "test-llm-primary",
            "start_time": time.time(),
            "pending_decision": None,
            "decision_result": None,
            "final_output": None,
            "error": None,
            "status": "running",
            "_async_task_id": "",
            "_task_queue_ref": None,
            "_llm_models": {},  # populated per-test below
            "enable_reflection": False,
        }

    @pytest.mark.asyncio
    async def test_simple_complexity_calls_llm(self, base_state):
        """Regression: simple complexity 应该调用 LLM，不跳过"""
        mock_llm = AsyncMock()
        mock_llm.model = "test-model"
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "goal": "simple blog",
                "reasoning": "simple plan",
                "tasks": [
                    {"agent_type": "WritingAgent", "input": {"topic": "test"}, "dependencies": []},
                    {"agent_type": "ReviewAgent", "input": {"topic": "test"}, "dependencies": [0]},
                ],
            })
        ))
        # _get_llm looks in _llm_models keyed by agent_type, not _manager_llm
        state = {**base_state, "_llm_models": {"ManagerAgent": mock_llm}}

        result = await classify_and_decompose(state)

        mock_llm.ainvoke.assert_called_once()
        assert result["plan"].get("plan_method") == "llm"
        assert result["status"] == "running"
        assert len(result["task_queue"]) == 2

    @pytest.mark.asyncio
    async def test_template_fallback_when_llm_raises(self, base_state):
        """LLM 抛异常时回退到模板"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM unavailable"))
        state = {**base_state, "_llm_models": {"ManagerAgent": mock_llm}}

        result = await classify_and_decompose(state)

        # Fallback returns dict with plan_method at result["plan"]["plan_method"]
        assert result["plan"].get("plan_method") == "template"
        assert result["status"] == "running"
        assert len(result["task_queue"]) >= 1

    @pytest.mark.asyncio
    async def test_template_fallback_when_no_llm(self, base_state):
        """没有 LLM 时使用模板"""
        state = {**base_state}

        result = await classify_and_decompose(state)

        assert result["plan"].get("plan_method") == "template"
        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_llm_plan_too_many_tasks_uses_template(self, base_state):
        """LLM 返回过多任务时回退到模板"""
        # Use a longer request so classify_complexity returns "medium"
        base_state["user_request"] = "帮我写一篇关于Python编程的深度技术教程，需要SEO优化，发布到微信公众号和知乎平台"
        mock_llm = AsyncMock()
        mock_llm.model = "test-model"
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "goal": "medium",
                "reasoning": "bad plan",
                "tasks": [
                    {"agent_type": "WritingAgent", "input": {"topic": "test"}, "dependencies": []},
                    {"agent_type": "ReviewAgent", "input": {"topic": "test"}, "dependencies": [0]},
                    {"agent_type": "SEOAgent", "input": {"topic": "test"}, "dependencies": [1]},
                    {"agent_type": "WebSearch", "input": {"query": "test"}, "dependencies": []},
                    {"agent_type": "DiagramAgent", "input": {"topic": "test"}, "dependencies": []},
                    {"agent_type": "CodeAgent", "input": {"topic": "test"}, "dependencies": []},
                    {"agent_type": "HotTopicAgent", "input": {}, "dependencies": []},
                    {"agent_type": "TranslationAgent", "input": {}, "dependencies": []},
                    {"agent_type": "ImageAgent", "input": {}, "dependencies": []},
                    {"agent_type": "PlatformAdapter", "input": {}, "dependencies": []},
                    {"agent_type": "WritingAgent", "input": {}, "dependencies": []},
                ],
            })
        ))
        state = {**base_state, "_llm_models": {"ManagerAgent": mock_llm}}

        result = await classify_and_decompose(state)

        assert result["plan"].get("plan_method") == "template"  # 回退，因为 > 8 个任务

    @pytest.mark.asyncio
    async def test_precomputed_plan_skips_llm(self, base_state):
        """有预计算计划时跳过 LLM"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock()
        precomputed = {
            "goal": "precomputed",
            "reasoning": "from upstream",
            "tasks": [
                {"id": "pre-1", "agent_type": "WritingAgent", "input": {}, "dependencies": []},
            ],
        }
        state = {**base_state, "_llm_models": {"ManagerAgent": mock_llm}, "plan": precomputed}

        result = await classify_and_decompose(state)

        mock_llm.ainvoke.assert_not_called()
        assert len(result["task_queue"]) == 1


# ══════════════════════════════════════════════
# Orchestrator _decompose — LLM primary 测试
# ══════════════════════════════════════════════

class TestOrchestratorLLMPrimary:
    """验证 Orchestrator._decompose 也使用 LLM 主路径"""

    @pytest.mark.asyncio
    async def test_decompose_simple_uses_llm(self):
        """Orchestrator._decompose 对 simple 复杂度也用 LLM"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "goal": "simple blog",
                "reasoning": "dynamic plan",
                "tasks": [
                    {"agent_type": "WritingAgent", "input": {"topic": "test"}, "dependencies": []},
                    {"agent_type": "ReviewAgent", "input": {"topic": "test"}, "dependencies": [0]},
                ],
            })
        ))
        orch = Orchestrator(chat_models={"ManagerAgent": mock_llm})

        result = await orch._decompose("写博客", {})

        mock_llm.ainvoke.assert_called_once()
        assert result.plan_method == "llm"
        assert len(result.tasks) == 2

    @pytest.mark.asyncio
    async def test_decompose_fallback_on_error(self):
        """Orchestrator._decompose 在 LLM 失败时回退"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("fail"))
        orch = Orchestrator(chat_models={"ManagerAgent": mock_llm})

        result = await orch._decompose("写博客", {})

        assert result.plan_method == "template"

    @pytest.mark.asyncio
    async def test_decompose_force_template_skips_llm(self):
        """force_template 参数跳过 LLM"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock()
        orch = Orchestrator(chat_models={"ManagerAgent": mock_llm})

        result = await orch._decompose("写博客", {"force_template": "simple"})

        mock_llm.ainvoke.assert_not_called()
        assert result.plan_method == "template"
        assert len(result.tasks) == 2


# ══════════════════════════════════════════════
# MessageBus — agent-initiated messaging 测试
# ══════════════════════════════════════════════

class TestMessageBusAgentMessaging:
    """测试新增的 agent-initiated 通信方法"""

    @pytest.mark.asyncio
    async def test_peer_feedback_sent(self):
        bus = MessageBus()

        await bus.peer_feedback(
            from_agent="ReviewAgent",
            to_agent="WritingAgent",
            feedback_content={"type": "revision_needed", "issues": ["too short"]},
            related_task_id="task-123",
        )

        pending = bus.get_pending("WritingAgent")
        assert len(pending) == 1
        assert pending[0].from_agent == "ReviewAgent"
        assert pending[0].content["type"] == "revision_needed"
        assert pending[0].metadata["related_task_id"] == "task-123"

    @pytest.mark.asyncio
    async def test_peer_feedback_no_reply_expected(self):
        """peer_feedback 是 fire-and-forget，不需要等待回复"""
        bus = MessageBus()
        await bus.peer_feedback(
            from_agent="WritingAgent",
            to_agent="SEOAgent",
            feedback_content={"type": "content_ready"},
        )
        # 不应该设置任何 reply event
        assert len(bus._reply_events) == 0

    def test_agent_discovery_sync(self):
        """agent_discovery uses registry to find matching agent"""
        from src.agent.agents import PREDEFINED_PROFILES

        mock_registry = MagicMock()
        mock_registry.find_by_expertise = MagicMock(return_value=[PREDEFINED_PROFILES["WritingAgent"]])

        bus = MessageBus(registry=mock_registry)

        result = bus.agent_discovery_sync("WritingAgent", "content creation")

        mock_registry.find_by_expertise.assert_called_once_with("content creation")
        assert result == "WritingAgent"

    def test_agent_discovery_no_match(self):
        bus = MessageBus()
        result = bus.agent_discovery_sync("ReviewAgent", "completely unknown skill")
        assert result is None

    @pytest.mark.asyncio
    async def test_agent_discovery_async(self):
        from src.agent.agents import PREDEFINED_PROFILES

        mock_registry = MagicMock()
        mock_registry.find_by_expertise = MagicMock(return_value=[PREDEFINED_PROFILES["SEOAgent"]])

        bus = MessageBus(registry=mock_registry)
        result = await bus.agent_discovery(requester="ManagerAgent", capability_needed="keyword research")

        assert result == "SEOAgent"


# ══════════════════════════════════════════════
# AgentRegistry — capability-based matching 测试
# ══════════════════════════════════════════════

class TestAgentRegistryCapability:
    """验证 AgentRegistry 能按能力匹配 Agent"""

    def test_find_by_expertise_existing(self):
        from src.agent.agents.registry import AgentRegistry

        registry = AgentRegistry()
        results = registry.find_by_expertise("SEO")

        assert len(results) >= 1
        assert any(p.name == "SEOAgent" for p in results)

    def test_find_by_expertise_no_match(self):
        from src.agent.agents.registry import AgentRegistry

        registry = AgentRegistry()
        results = registry.find_by_expertise("xyzzy_nomatch")

        assert len(results) == 0


# ══════════════════════════════════════════════
# Schema — 新字段测试
# ══════════════════════════════════════════════

class TestSchemaNewFields:
    """验证 AgentTask 和 ExecutionPlan 新字段"""

    def test_agent_task_new_fields(self):
        task = AgentTask(
            id="t1",
            agent_type="WritingAgent",
            input={"topic": "test"},
            can_parallel_with=["t2"],
            retry_on_low_quality=True,
            priority="high",
        )
        assert task.can_parallel_with == ["t2"]
        assert task.retry_on_low_quality is True
        assert task.priority == "high"

    def test_execution_plan_new_fields(self):
        plan = ExecutionPlan(
            goal="test goal",
            tasks=[],
            reasoning="test",
            estimated_steps=2,
            parallel_groups=[["a", "b"], ["c"]],
            plan_method="llm",
        )
        assert plan.parallel_groups == [["a", "b"], ["c"]]
        assert plan.plan_method == "llm"

    def test_agent_task_defaults(self):
        """新字段有默认值，不破坏现有代码"""
        task = AgentTask(
            id="t1",
            agent_type="WritingAgent",
            input={},
        )
        assert task.can_parallel_with == []
        assert task.retry_on_low_quality is False
        assert task.priority == "medium"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
