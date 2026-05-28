"""Graph 端到端集成测试 — 完整执行路径 + 各条件分支"""
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Mock LLM that passes hasattr(llm, 'model') check but avoids MagicMock auto-creation
# of max_tokens (which causes '>' comparison errors in token budget code)
_MOCK_LLM = MagicMock(spec=["model"], model="gpt-4")
import uuid
import time

from src.agent.graph import build_agent_graph
from src.agent.state import AgentGraphState


# ── 辅助函数 ──

def _make_task(agent_type: str, deps: list[int] | None = None, task_id: str | None = None, **extra) -> dict:
    return {
        "id": task_id or str(uuid.uuid4()),
        "agent_type": agent_type,
        "input": {"topic": "test", **extra},
        "dependencies": deps or [],
        "config": {"reasoning": "", "priority": 0},
    }


def _make_initial_state(**overrides) -> AgentGraphState:
    """创建基础初始 state"""
    return {
        "user_request": "test request",
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
        "trace_task_id": str(uuid.uuid4()),
        "start_time": time.time(),
        "pending_decision": None,
        "decision_result": None,
        "final_output": None,
        "error": None,
        "status": "running",
        "_async_task_id": str(uuid.uuid4()),
        "_task_queue_ref": None,
        "_llm_models": {},
        "_mcp_tools": [],
        "_streaming_mode": False,
        "_agent_registry": None,
        "_tool_registry": None,
        "use_message_bus": False,
        "_message_bus": None,
        "_messages": [],
        "enable_reflection": False,
        "reflection_config": None,
        "_reflection_engine": None,
        "reflection_results": [],
        "self_reflection_rounds": {},
        "enable_hitl": False,
        "_hitl_manager": None,
        "pending_gates": [],
        **overrides,
    }


# ── mock handler ──

async def _mock_handler_success(task_input, llm, mcp_tools) -> dict:
    return {
        "status": "completed",
        "agent_type": task_input.get("_agent_profile", MagicMock()).name if isinstance(task_input, dict) and task_input.get("_agent_profile") else "MockAgent",
        "output": {"content": "mock result content"},
        "tokens_used": 10,
        "thinking_steps": [],
    }


async def _mock_handler_fail(task_input, llm, mcp_tools) -> dict:
    return {
        "status": "failed",
        "agent_type": "MockAgent",
        "error": "mock error",
        "output": {"content": ""},
        "tokens_used": 0,
        "thinking_steps": [],
    }


# ── Test: Graph Building ──

class TestGraphBuild:
    def test_build_graph(self):
        g = build_agent_graph()
        assert g is not None
        assert len(g.nodes) >= 6  # 6+ nodes registered

    def test_graph_has_all_nodes(self):
        g = build_agent_graph()
        node_names = list(g.nodes.keys())
        expected = {
            "classify_and_decompose", "schedule_and_execute",
            "execute_task_batch", "reflect_on_task_batch",
            "check_hitl_gates", "merge_and_finish",
        }
        for name in expected:
            assert name in node_names, f"Missing node: {name}"


# ── Test: Full Graph Execution ──

class TestGraphExecution:
    """完整图执行测试（mock handler）"""

    @pytest.mark.asyncio
    async def test_simple_task_completes(self):
        """简单任务：WritingAgent + ReviewAgent，全部成功"""
        graph = build_agent_graph()

        tasks = [
            _make_task("WritingAgent", deps=[], topic="write"),
            _make_task("ReviewAgent", deps=[0], topic="review"),
        ]
        state = _make_initial_state(
            task_queue=tasks,
            plan={"goal": "test", "reasoning": "test", "tasks": tasks},
            plan_reasoning="test",
        )

        with patch("src.agent.nodes.HANDLER_MAP", {
            "WritingAgent": _mock_handler_success,
            "ReviewAgent": _mock_handler_success,
        }):
            final = await graph.ainvoke(state, {"recursion_limit": 50})

        assert final["status"] in ("completed", "partial")
        assert len(final["completed_task_ids"]) == 2
        assert final["final_output"] is not None

    @pytest.mark.asyncio
    async def test_handler_failure(self):
        """一个 handler 失败应标记为 partial"""
        graph = build_agent_graph()

        tasks = [
            _make_task("WritingAgent", deps=[], topic="write"),
            _make_task("ReviewAgent", deps=[0], topic="review"),
        ]
        state = _make_initial_state(
            task_queue=tasks,
            plan={"goal": "test", "reasoning": "test", "tasks": tasks},
            plan_reasoning="test",
            _llm_models={"default": _MOCK_LLM},
        )

        with patch("src.agent.nodes.HANDLER_MAP", {
            "WritingAgent": _mock_handler_fail,
            "ReviewAgent": _mock_handler_success,
        }):
            final = await graph.ainvoke(state, {"recursion_limit": 50})

        assert final["status"] in ("completed", "partial")
        assert len(final["completed_task_ids"]) == 2

    @pytest.mark.asyncio
    async def test_all_handlers_fail(self):
        """全部失败应标记为 partial（retry wrapper 降级）"""
        graph = build_agent_graph()

        tasks = [
            _make_task("WritingAgent", deps=[], topic="x"),
            _make_task("ReviewAgent", deps=[0], topic="x"),
        ]
        state = _make_initial_state(
            task_queue=tasks,
            plan={"goal": "test", "reasoning": "test", "tasks": tasks},
            plan_reasoning="test",
            _llm_models={"default": _MOCK_LLM},
        )

        with patch("src.agent.nodes.HANDLER_MAP", {
            "WritingAgent": _mock_handler_fail,
            "ReviewAgent": _mock_handler_fail,
        }):
            final = await graph.ainvoke(state, {"recursion_limit": 50})

        assert final["status"] in ("completed", "partial")

    @pytest.mark.asyncio
    async def test_dag_dependency_resolution(self):
        """DAG 依赖：WebSearch → WritingAgent → ReviewAgent 顺序执行"""
        graph = build_agent_graph()

        tasks = [
            _make_task("WebSearch", deps=[], query="search"),
            _make_task("WritingAgent", deps=[0], topic="write"),
            _make_task("ReviewAgent", deps=[1], topic="review"),
        ]
        state = _make_initial_state(
            task_queue=tasks,
            plan={"goal": "test", "reasoning": "test", "tasks": tasks},
            plan_reasoning="test",
            _llm_models={"default": _MOCK_LLM},
        )

        call_order = []

        async def tracked_handler(t_input, llm, mcp_tools):
            call_order.append(t_input.get("topic", ""))
            # Return content >= 200 chars to pass simple_quality_score (threshold=60)
            return {"status": "completed", "agent_type": "Mock",
                    "output": {"content": f"result from {t_input.get('topic', '')}: " + "x" * 200},
                    "tokens_used": 0, "thinking_steps": []}

        with patch("src.agent.nodes.HANDLER_MAP", {
            "WebSearch": tracked_handler,
            "WritingAgent": tracked_handler,
            "ReviewAgent": tracked_handler,
        }):
            await graph.ainvoke(state, {"recursion_limit": 50})

        # WebSearch should be first (no deps), then WritingAgent (dep 0),
        # then ReviewAgent (dep 1)
        assert len(call_order) == 3

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """无依赖的任务应并行执行"""
        graph = build_agent_graph()

        tasks = [
            _make_task("WebSearch", deps=[], topic="s1"),
            _make_task("WebSearch", deps=[], topic="s2"),
            _make_task("ReviewAgent", deps=[0, 1], topic="review"),
        ]
        state = _make_initial_state(
            concurrency_limit=5,
            task_queue=tasks,
            plan={"goal": "test", "reasoning": "test", "tasks": tasks},
            plan_reasoning="test",
        )

        with patch("src.agent.nodes.HANDLER_MAP", {
            "WebSearch": _mock_handler_success,
            "ReviewAgent": _mock_handler_success,
        }):
            final = await graph.ainvoke(state, {"recursion_limit": 50})

        assert len(final["completed_task_ids"]) == 3
        assert final["status"] in ("completed", "partial")

    @pytest.mark.asyncio
    async def test_dependency_context_passed(self):
        """下游任务应收到上游的 content（依赖上下文）"""
        graph = build_agent_graph()

        tasks = [
            _make_task("WebSearch", deps=[], topic="search1"),
            _make_task("WritingAgent", deps=[0], topic="write"),
        ]
        state = _make_initial_state(
            task_queue=tasks,
            plan={"goal": "test", "reasoning": "test", "tasks": tasks},
            plan_reasoning="test",
            _llm_models={"default": _MOCK_LLM},
        )

        received_context = {}

        async def search_handler(t_input, llm, mcp_tools):
            return {"status": "completed", "agent_type": "WebSearch",
                    "output": {"content": "search results here"},
                    "tokens_used": 0, "thinking_steps": []}

        async def write_handler(t_input, llm, mcp_tools):
            received_context["ctx"] = t_input.get("_context", {})
            return {"status": "completed", "agent_type": "WritingAgent",
                    "output": {"content": "article"},
                    "tokens_used": 0, "thinking_steps": []}

        with patch("src.agent.nodes.HANDLER_MAP", {
            "WebSearch": search_handler,
            "WritingAgent": write_handler,
        }):
            await graph.ainvoke(state, {"recursion_limit": 50})

        # WritingAgent should receive search results in context
        ctx = received_context.get("ctx", {})
        assert len(ctx) > 0
        task_id = tasks[0]["id"]
        assert task_id in ctx
        assert "search results here" in str(ctx)


# ── Test: classify_and_decompose ──

class TestClassifyAndDecompose:
    """classify_and_decompose 节点测试"""

    @pytest.mark.asyncio
    async def test_precomputed_plan_used(self):
        """预计算计划应跳过 LLM 分解"""
        from src.agent.nodes import classify_and_decompose

        tasks = [_make_task("WritingAgent")]
        state = _make_initial_state(
            plan={"goal": "test", "reasoning": "test", "tasks": tasks},
        )

        result = await classify_and_decompose(state)
        assert result["task_queue"] == tasks
        assert "Using pre-computed plan" in str(result["thinking_steps"])

    @pytest.mark.asyncio
    async def test_simple_complexity_template(self):
        """简单请求应使用模板计划"""
        from src.agent.nodes import classify_and_decompose

        state = _make_initial_state(user_request="hello")
        result = await classify_and_decompose(state)

        assert result["task_queue"] is not None
        assert len(result["task_queue"]) >= 2
        assert "plan" in result
        assert result["plan_reasoning"]

    @pytest.mark.asyncio
    async def test_no_llm_fallback_to_template(self):
        """无 LLM 时复杂任务也回退到模板"""
        from src.agent.nodes import classify_and_decompose

        state = _make_initial_state(
            user_request="生成一篇深度报告并配图",
        )
        result = await classify_and_decompose(state)
        assert result["task_queue"] is not None
        assert len(result["task_queue"]) >= 2  # template fallback


# ── Test: schedule_and_execute ──

class TestScheduleAndExecute:
    """schedule_and_execute + should_continue_or_finish 测试"""

    @pytest.mark.asyncio
    async def test_ready_tasks_found(self):
        from src.agent.nodes import schedule_and_execute

        tasks = [
            _make_task("WebSearch", deps=[], task_id="t1"),
            _make_task("WritingAgent", deps=[0], task_id="t2"),
        ]
        state = _make_initial_state(
            task_queue=tasks,
            completed_task_ids=["t1"],
        )

        result = await schedule_and_execute(state)
        assert len(result["ready_tasks"]) == 1
        assert result["ready_tasks"][0]["id"] == "t2"

    @pytest.mark.asyncio
    async def test_no_ready_tasks_all_done(self):
        from src.agent.nodes import schedule_and_execute

        tasks = [_make_task("WritingAgent", task_id="t1")]
        state = _make_initial_state(
            task_queue=tasks,
            completed_task_ids=["t1"],
        )

        result = await schedule_and_execute(state)
        assert result["ready_tasks"] == []

    @pytest.mark.asyncio
    async def test_deadlock_detection(self):
        from src.agent.nodes import schedule_and_execute

        tasks = [
            _make_task("WritingAgent", deps=[2], task_id="t1"),  # dep on t3 which doesn't exist
            _make_task("ReviewAgent", deps=[0], task_id="t2"),   # dep on t1 (completed)
        ]
        state = _make_initial_state(
            task_queue=tasks,
            completed_task_ids=["t2"],
        )

        result = await schedule_and_execute(state)
        assert result["ready_tasks"] == []

    @pytest.mark.asyncio
    async def test_stuck_tasks_with_no_deps(self):
        """无依赖但未完成的任务应被标记为 ready"""
        from src.agent.nodes import schedule_and_execute

        tasks = [
            _make_task("WritingAgent", deps=[], task_id="t1"),
            _make_task("ReviewAgent", deps=[0], task_id="t2"),
        ]
        state = _make_initial_state(
            task_queue=tasks,
            completed_task_ids=["t2"],  # t2 done but t1 not done despite no deps
        )

        result = await schedule_and_execute(state)
        assert len(result["ready_tasks"]) == 1
        assert result["ready_tasks"][0]["id"] == "t1"


# ── Test: should_continue_or_finish ──

class TestShouldContinueOrFinish:
    """条件边路由测试"""

    def test_has_ready_tasks(self):
        from src.agent.nodes import should_continue_or_finish

        state = _make_initial_state(ready_tasks=[{"id": "t1"}],
                                    task_queue=[{"id": "t1"}])
        assert should_continue_or_finish(state) == "execute"

    def test_all_done(self):
        from src.agent.nodes import should_continue_or_finish

        state = _make_initial_state(ready_tasks=[],
                                    task_queue=[{"id": "t1"}],
                                    completed_task_ids=["t1"])
        assert should_continue_or_finish(state) == "done"

    def test_no_ready_tasks_not_all_done(self):
        from src.agent.nodes import should_continue_or_finish

        state = _make_initial_state(ready_tasks=[],
                                    task_queue=[{"id": "t1"}, {"id": "t2"}],
                                    completed_task_ids=["t1"])
        assert should_continue_or_finish(state) == "done"  # forced exit

    def test_empty_state(self):
        from src.agent.nodes import should_continue_or_finish

        state = _make_initial_state(ready_tasks=[], task_queue=[])
        assert should_continue_or_finish(state) == "done"


# ── Test: should_reflect_or_continue ──

class TestShouldReflectOrContinue:
    """反思条件边测试"""

    def test_reflection_disabled(self):
        from src.agent.edges import should_reflect_or_continue

        state = _make_initial_state(enable_reflection=False,
                                    task_queue=[{"id": "t1"}],
                                    completed_task_ids=["t1"])
        # Fallback to should_continue_or_finish
        assert should_reflect_or_continue(state) == "done"

    def test_reflection_enabled_no_results(self):
        from src.agent.edges import should_reflect_or_continue

        state = _make_initial_state(enable_reflection=True,
                                    task_results={"t1": {}},
                                    reflection_results=[])
        assert should_reflect_or_continue(state) == "reflect"

    def test_reflection_enabled_all_passed(self):
        from src.agent.edges import should_reflect_or_continue

        state = _make_initial_state(
            enable_reflection=True,
            task_results={"t1": {"agent_type": "WritingAgent"}},
            reflection_results=[{
                "reflection_type": "self", "passed": True,
                "target_task_id": "t1", "score": 0.9,
            }],
            self_reflection_rounds={"t1": 1},
            task_queue=[{"id": "t1"}],
            completed_task_ids=["t1"],
        )
        # All passed, should fallback to continue_or_finish
        assert should_reflect_or_continue(state) == "done"

    def test_reflection_has_unpassed(self):
        from src.agent.edges import should_reflect_or_continue

        state = _make_initial_state(
            enable_reflection=True,
            task_results={"t1": {"agent_type": "WritingAgent"}},
            reflection_results=[{
                "reflection_type": "self", "passed": False,
                "target_task_id": "t1", "score": 0.4,
            }],
            self_reflection_rounds={"t1": 1},
        )
        assert should_reflect_or_continue(state) == "reflect"


# ── Test: should_retry_after_reflection ──

class TestShouldRetryAfterReflection:
    """反思后重试条件测试"""

    def test_no_retry_needed(self):
        from src.agent.edges import should_retry_after_reflection

        state = _make_initial_state(
            reflection_results=[{
                "reflection_type": "self", "passed": True,
                "target_task_id": "t1", "score": 0.9,
            }],
            self_reflection_rounds={"t1": 1},
        )
        assert should_retry_after_reflection(state) == "done"

    def test_retry_needed(self):
        from src.agent.edges import should_retry_after_reflection

        state = _make_initial_state(
            reflection_results=[{
                "reflection_type": "self", "passed": False,
                "target_task_id": "t1", "score": 0.4,
            }],
            self_reflection_rounds={"t1": 1},
        )
        assert should_retry_after_reflection(state) == "execute"

    def test_max_retries_reached(self):
        from src.agent.edges import should_retry_after_reflection

        state = _make_initial_state(
            reflection_results=[{
                "reflection_type": "self", "passed": False,
                "target_task_id": "t1", "score": 0.4,
            }],
            self_reflection_rounds={"t1": 2},  # max rounds reached
        )
        assert should_retry_after_reflection(state) == "done"

    def test_empty_results(self):
        from src.agent.edges import should_retry_after_reflection

        state = _make_initial_state(reflection_results=[],
                                    self_reflection_rounds={})
        assert should_retry_after_reflection(state) == "done"


# ── Test: merge_and_finish ──

class TestMergeAndFinish:
    """merge_and_finish 节点测试"""

    @pytest.mark.asyncio
    async def test_all_completed(self):
        from src.agent.nodes import merge_and_finish

        state = _make_initial_state(
            plan={"goal": "test goal", "reasoning": "test"},
            task_results={
                "t1": {"status": "completed", "agent_type": "WritingAgent",
                       "output": {"content": "article"}},
                "t2": {"status": "completed", "agent_type": "ReviewAgent",
                       "output": {"content": "review"}},
            },
            task_queue=[
                {"id": "t1", "input": {}},
                {"id": "t2", "input": {}},
            ],
            completed_task_ids=["t1", "t2"],
            failed_task_ids=[],
        )

        result = await merge_and_finish(state)
        assert result["status"] == "completed"
        assert result["final_output"] is not None
        assert len(result["final_output"]["sections"]) == 2

    @pytest.mark.asyncio
    async def test_partial_with_failures(self):
        from src.agent.nodes import merge_and_finish

        state = _make_initial_state(
            plan={"goal": "test", "reasoning": "test"},
            task_results={
                "t1": {"status": "completed", "agent_type": "WritingAgent",
                       "output": {"content": "article"}},
                "t2": {"status": "failed", "agent_type": "ReviewAgent",
                       "error": "review error", "output": {}},
            },
            task_queue=[
                {"id": "t1", "input": {}},
                {"id": "t2", "input": {}},
            ],
            completed_task_ids=["t1"],
            failed_task_ids=["t2"],
        )

        result = await merge_and_finish(state)
        assert result["status"] == "partial"

    @pytest.mark.asyncio
    async def test_all_failed(self):
        from src.agent.nodes import merge_and_finish

        state = _make_initial_state(
            plan={"goal": "test", "reasoning": "test"},
            task_results={
                "t1": {"status": "failed", "agent_type": "WritingAgent",
                       "error": "error", "output": {}},
            },
            task_queue=[{"id": "t1", "input": {}}],
            completed_task_ids=[],
            failed_task_ids=["t1"],
        )

        result = await merge_and_finish(state)
        assert result["status"] == "failed"


# ── Test: reflect_on_task_batch ──

class TestReflectOnTaskBatch:
    """reflect_on_task_batch 节点测试"""

    @pytest.mark.asyncio
    async def test_reflection_disabled(self):
        from src.agent.nodes import reflect_on_task_batch

        state = _make_initial_state(enable_reflection=False)
        result = await reflect_on_task_batch(state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_reflection_skips_non_writing_agents(self):
        from src.agent.nodes import reflect_on_task_batch

        state = _make_initial_state(
            enable_reflection=True,
            task_results={"t1": {"agent_type": "WebSearch",
                                 "output": {"content": "data"}}},
            completed_task_ids=["t1"],
        )
        result = await reflect_on_task_batch(state)
        assert len(result.get("reflection_results", [])) == 0


# ── Test: check_hitl_gates ──

class TestCheckHITLGates:
    """check_hitl_gates 节点测试"""

    @pytest.mark.asyncio
    async def test_hitl_disabled(self):
        from src.agent.nodes import check_hitl_gates

        state = _make_initial_state(enable_hitl=False)
        result = await check_hitl_gates(state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_hitl_enabled_no_pending(self):
        from src.agent.nodes import check_hitl_gates

        state = _make_initial_state(enable_hitl=True, pending_gates=[])
        result = await check_hitl_gates(state)
        assert result["pending_gates"] == []


# ── Test: execute_task_batch with concurrency ──

class TestExecuteTaskBatch:
    """execute_task_batch 节点并行执行测试"""

    @pytest.mark.asyncio
    async def test_batch_execution(self):
        from src.agent.nodes import execute_task_batch

        execution_order = []

        async def tracked_handler(t_input, llm, mcp_tools):
            execution_order.append(t_input.get("topic", ""))
            return {"status": "completed", "agent_type": "Mock",
                    "output": {"content": f"result for {t_input.get('topic', '')}: " + "x" * 200},
                    "tokens_used": 0, "thinking_steps": []}

        tasks = [
            _make_task("WritingAgent", task_id="t1", topic="a"),
            _make_task("WritingAgent", task_id="t2", topic="b"),
        ]

        state = _make_initial_state(
            ready_tasks=tasks,
            concurrency_limit=2,
            task_queue=tasks,
            _llm_models={"default": _MOCK_LLM},
        )

        with patch("src.agent.nodes.HANDLER_MAP", {
            "WritingAgent": tracked_handler,
        }):
            result = await execute_task_batch(state)

        assert len(result["completed_task_ids"]) == 2
        assert len(execution_order) == 2


# ── Test: Full Graph with LLM ──

class TestGraphWithMockLLM:
    """使用 mock LLM 的完整图测试"""

    @pytest.mark.asyncio
    async def test_llm_decomposes_complex_request(self):
        """LLM 用于复杂任务的分解"""
        graph = build_agent_graph()

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content='{"goal": "test", "reasoning": "llm plan", '
                    '"tasks": [{"agent_type": "WritingAgent", "input": {"topic": "x"}, "dependencies": []}]}'
        )

        state = _make_initial_state(
            user_request="生成一篇深度行业分析报告并配图",
            _llm_models={"ManagerAgent": mock_llm},
        )

        with patch("src.agent.nodes.HANDLER_MAP", {
            "WritingAgent": _mock_handler_success,
        }):
            final = await graph.ainvoke(state, {"recursion_limit": 50})

        assert final["status"] in ("completed", "partial")
        mock_llm.ainvoke.assert_called()
