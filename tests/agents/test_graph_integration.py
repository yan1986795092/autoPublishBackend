"""Graph 集成测试 — 仅测试当前简化后的节点"""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain_openai import ChatOpenAI

from src.agent.graph import build_agent_graph
from src.agent.state import AgentGraphState
from src.agent.nodes import (
    classify_and_decompose,
    schedule_and_execute,
    execute_task_batch,
    merge_and_finish,
    should_continue_or_finish,
    _build_fallback_plan,
    _normalize_plan_for_context,
)

# ── Mock helpers ──

_MOCK_LLM = MagicMock(spec=ChatOpenAI)
_MOCK_LLM.model_name = "mock-model"
_MOCK_LLM.max_tokens = 4000

_MOCK_LLM_RESPONSE = MagicMock()
_MOCK_LLM_RESPONSE.content = '{"goal":"test","tasks":[{"agent_type":"WritingAgent","input":{"topic":"write"}}],"reasoning":"test"}'
_MOCK_LLM.ainvoke = AsyncMock(return_value=_MOCK_LLM_RESPONSE)


async def _mock_handler_success(task_input: dict, llm, mcp_tools=None) -> dict:
    return {
        "status": "completed",
        "agent_type": task_input.get("_task_id", "WritingAgent"),
        "output": {"content": "mock output", "type": "text"},
        "tokens_used": 10,
        "thinking_steps": [],
    }


async def _mock_handler_fail(task_input: dict, llm, mcp_tools=None) -> dict:
    return {
        "status": "failed",
        "agent_type": task_input.get("_task_id", "WritingAgent"),
        "error": "mock error",
        "tokens_used": 0,
        "thinking_steps": [],
    }


def _make_task(agent_type: str, deps: list | None = None, topic: str = "test") -> dict:
    import uuid
    return {
        "id": str(uuid.uuid4()),
        "agent_type": agent_type,
        "input": {"topic": topic, "content_type": "blog"},
        "dependencies": deps or [],
        "can_parallel_with": [],
        "retry_on_low_quality": False,
        "priority": "medium",
        "config": {},
    }


def _make_initial_state(**overrides) -> AgentGraphState:
    return {
        "user_request": "test request",
        "context": {},
        "skill_system_prompt": "",
        "skill_name": "",
        "plan": None,
        "plan_reasoning": "",
        "task_queue": [],
        "completed_task_ids": [],
        "failed_task_ids": [],
        "task_results": {},
        "ready_tasks": [],
        "concurrency_limit": 3,
        "thinking_steps": [],
        "trace_task_id": "test-trace",
        "start_time": 0.0,
        "final_output": None,
        "error": None,
        "status": "running",
        "_async_task_id": "test-async",
        "_task_queue_ref": None,
        "_llm_models": {"default": _MOCK_LLM},
        "_mcp_tools": [],
        "_streaming_mode": False,
        **overrides,
    }


# ── Test: Graph Building ──

class TestGraphBuild:
    def test_build_graph(self):
        g = build_agent_graph()
        assert g is not None
        assert len(g.nodes) >= 4

    def test_graph_has_all_nodes(self):
        g = build_agent_graph()
        node_names = list(g.nodes.keys())
        expected = {
            "classify_and_decompose", "schedule_and_execute",
            "execute_task_batch", "merge_and_finish",
        }
        for name in expected:
            assert name in node_names, f"Missing node: {name}"


# ── Test: Full Graph Execution ──

class TestGraphExecution:
    @pytest.mark.asyncio
    async def test_simple_task_completes(self):
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
        assert final["final_output"] is not None

    @pytest.mark.asyncio
    async def test_dag_dependency_resolution(self):
        graph = build_agent_graph()
        tasks = [
            _make_task("WebSearch", deps=[], topic="search"),
            _make_task("WritingAgent", deps=[0], topic="write"),
            _make_task("ReviewAgent", deps=[1], topic="review"),
        ]
        state = _make_initial_state(
            task_queue=tasks,
            plan={"goal": "test", "reasoning": "test", "tasks": tasks},
            plan_reasoning="test",
        )
        with patch("src.agent.nodes.HANDLER_MAP", {
            "WebSearch": _mock_handler_success,
            "WritingAgent": _mock_handler_success,
            "ReviewAgent": _mock_handler_success,
        }):
            final = await graph.ainvoke(state, {"recursion_limit": 50})

        assert final["status"] in ("completed", "partial")
        assert len(final["completed_task_ids"]) == 3

    @pytest.mark.asyncio
    async def test_all_handlers_fail(self):
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

        assert final["status"] == "failed"


# ── Test: classify_and_decompose ──

class TestClassifyAndDecompose:
    @pytest.mark.asyncio
    async def test_precomputed_plan_used(self):
        tasks = [_make_task("WritingAgent")]
        state = _make_initial_state(
            plan={"goal": "test", "reasoning": "test", "tasks": tasks},
        )
        result = await classify_and_decompose(state)
        assert result["task_queue"] == tasks
        assert "预计算" in str(result["thinking_steps"])

    @pytest.mark.asyncio
    async def test_simple_complexity(self):
        state = _make_initial_state(user_request="写一篇博客文章")
        result = await classify_and_decompose(state)
        assert "task_queue" in result

    @pytest.mark.asyncio
    async def test_no_llm_fallback(self):
        state = _make_initial_state(
            user_request="test",
            _llm_models={},
        )
        result = await classify_and_decompose(state)
        assert "task_queue" in result

    def test_normalize_revise_plan_removes_writing_agent(self):
        plan = {
            "goal": "revise content",
            "tasks": [
                _make_task("WritingAgent", deps=[], topic="draft"),
                _make_task("ReviewAgent", deps=[0], topic="review"),
                _make_task("FixAgent", deps=[0, 1], topic="fix"),
            ],
        }
        context = {
            "action": "revise",
            "topic": "润色这篇文章",
            "revision_focus": "开头更吸引人，压缩第二段",
            "content": "这是一段已有正文。",
        }

        normalized = _normalize_plan_for_context(plan, context, "润色这篇文章")
        agent_types = [task["agent_type"] for task in normalized["tasks"]]

        assert "WritingAgent" not in agent_types
        assert agent_types[0] == "ReviewAgent"
        assert agent_types[-1] == "FixAgent"
        assert normalized["tasks"][0]["input"]["content"] == "这是一段已有正文。"
        assert normalized["tasks"][-1]["input"]["feedback"] == "开头更吸引人，压缩第二段"

    def test_fallback_search_plan_has_full_pipeline(self):
        plan = _build_fallback_plan("搜索 2026 AI 热点趋势，并分析后写成文章")
        agent_types = [task["agent_type"] for task in plan["tasks"]]

        assert agent_types[:5] == [
            "WebSearch",
            "SearchAnalyzer",
            "WritingAgent",
            "ReviewAgent",
            "FixAgent",
        ]


# ── Test: should_continue_or_finish ──

class TestShouldContinueOrFinish:
    def test_ready_tasks_return_execute(self):
        state = _make_initial_state(ready_tasks=[_make_task("WritingAgent")])
        assert should_continue_or_finish(state) == "execute"

    def test_all_done_return_done(self):
        t = _make_task("WritingAgent")
        state = _make_initial_state(
            task_queue=[t],
            completed_task_ids=[t["id"]],
        )
        assert should_continue_or_finish(state) == "done"

    def test_empty_task_queue_return_done(self):
        state = _make_initial_state()
        assert should_continue_or_finish(state) == "done"


# ── Test: schedule_and_execute ──

class TestScheduleAndExecute:
    @pytest.mark.asyncio
    async def test_ready_tasks_found(self):
        t = _make_task("WritingAgent")
        state = _make_initial_state(task_queue=[t])
        result = await schedule_and_execute(state)
        assert len(result["ready_tasks"]) == 1

    @pytest.mark.asyncio
    async def test_dependency_not_met(self):
        t1 = _make_task("WritingAgent")
        t2 = _make_task("ReviewAgent", deps=[0])
        state = _make_initial_state(task_queue=[t1, t2])
        result = await schedule_and_execute(state)
        assert len(result["ready_tasks"]) == 1  # t1 only


# ── Test: execute_task_batch ──

class TestExecuteTaskBatch:
    @pytest.mark.asyncio
    async def test_batch_executes_all_ready(self):
        t = _make_task("WritingAgent")
        state = _make_initial_state(
            ready_tasks=[t],
            task_queue=[t],
        )
        with patch("src.agent.nodes.HANDLER_MAP", {"WritingAgent": _mock_handler_success}):
            result = await execute_task_batch(state)

        assert len(result["completed_task_ids"]) == 1
        assert t["id"] in result["completed_task_ids"]

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        state = _make_initial_state()
        result = await execute_task_batch(state)
        assert len(result["completed_task_ids"]) == 0
        assert len(result["failed_task_ids"]) == 0


# ── Test: merge_and_finish ──

class TestMergeAndFinish:
    @pytest.mark.asyncio
    async def test_merge_completed(self):
        t = _make_task("WritingAgent")
        state = _make_initial_state(
            task_queue=[t],
            completed_task_ids=[t["id"]],
            task_results={t["id"]: {
                "status": "completed",
                "agent_type": "WritingAgent",
                "output": {"content": "hello world", "content_type": "blog"},
                "tokens_used": 10,
            }},
            plan={"goal": "test goal"},
        )
        result = await merge_and_finish(state)
        assert result["status"] == "completed"
        assert result["final_output"] is not None
