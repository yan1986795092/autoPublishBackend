"""Orchestrator 单元测试：任务拆解、Graph 节点、回退计划"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import time

from src.agent.orchestrator import Orchestrator
from src.agent.schemas import AgentTask, ExecutionPlan, TaskStatus, UsageInfo
from src.agent.state import AgentGraphState
from src.agent.nodes import (
    classify_and_decompose,
    schedule_and_execute,
    merge_and_finish,
    should_continue_or_finish,
)


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.model = "test-model"
    llm.generate_sync = AsyncMock(return_value=json.dumps({
        "goal": "test goal",
        "reasoning": "because",
        "tasks": [
            {"agent_type": "WritingAgent", "input": {"topic": "test"}, "dependencies": []},
            {"agent_type": "ReviewAgent", "input": {"topic": "test"}, "dependencies": [0]},
        ],
    }))
    return llm


@pytest.fixture
def orchestrator(mock_llm):
    return Orchestrator(llm_gateway=mock_llm)


# ─────────────────────────────
# Graph Node 单元测试
# ─────────────────────────────


class TestGraphNodes:
    """直接测试 LangGraph 节点函数"""

    @pytest.mark.asyncio
    async def test_classify_and_decompose_simple(self):
        """简单请求 → classify 返回 continue + 2 个 task"""
        state: AgentGraphState = {
            "user_request": "写一篇关于AI的文章",
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
            "trace_task_id": "test-1",
            "start_time": time.time(),
            "pending_decision": None,
            "decision_result": None,
            "final_output": None,
            "error": None,
            "status": "running",
            "_async_task_id": "",
            "_task_queue_ref": None,
            "_llm_models": {},
        }
        result = await classify_and_decompose(state)
        assert result["status"] == "running"
        assert result["plan"] is not None
        assert len(result["task_queue"]) >= 1
        assert len(result["thinking_steps"]) >= 1

    @pytest.mark.asyncio
    async def test_classify_and_decompose_empty_request(self):
        """空请求也能生成 fallback 计划"""
        state: AgentGraphState = {
            "user_request": "",
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
            "trace_task_id": "test-2",
            "start_time": time.time(),
            "pending_decision": None,
            "decision_result": None,
            "final_output": None,
            "error": None,
            "status": "running",
            "_async_task_id": "",
            "_task_queue_ref": None,
            "_llm_models": {},
        }
        result = await classify_and_decompose(state)
        assert result["status"] == "running"
        assert len(result["task_queue"]) >= 1

    @pytest.mark.asyncio
    async def test_find_ready_tasks_all_independent(self):
        """所有任务无依赖 → 全部进入 ready_tasks"""
        state: AgentGraphState = {
            "user_request": "test",
            "context": {},
            "plan": None,
            "plan_reasoning": "",
            "task_queue": [
                {"id": "a", "agent_type": "WritingAgent", "dependencies": []},
                {"id": "b", "agent_type": "ReviewAgent", "dependencies": []},
            ],
            "completed_task_ids": [],
            "failed_task_ids": [],
            "task_results": {},
            "ready_tasks": [],
            "concurrency_limit": 3,
            "thinking_steps": [],
            "trace_task_id": "test-3",
            "start_time": time.time(),
            "pending_decision": None,
            "decision_result": None,
            "final_output": None,
            "error": None,
            "status": "running",
            "_async_task_id": "",
            "_task_queue_ref": None,
            "_llm_models": {},
        }
        result = await schedule_and_execute(state)
        assert len(result["ready_tasks"]) == 2

    @pytest.mark.asyncio
    async def test_find_ready_tasks_with_deps(self):
        """有依赖的任务在依赖未完成时不出现在 ready_tasks"""
        state: AgentGraphState = {
            "user_request": "test",
            "context": {},
            "plan": None,
            "plan_reasoning": "",
            "task_queue": [
                {"id": "a", "agent_type": "WritingAgent", "dependencies": []},
                {"id": "b", "agent_type": "ReviewAgent", "dependencies": ["a"]},
            ],
            "completed_task_ids": ["a"],
            "task_results": {
                "a": {"agent_type": "WritingAgent", "output": {"content": "done"}, "status": "completed"}
            },
            "failed_task_ids": [],
            "ready_tasks": [],
            "concurrency_limit": 3,
            "thinking_steps": [],
            "trace_task_id": "test-4",
            "start_time": time.time(),
            "pending_decision": None,
            "decision_result": None,
            "final_output": None,
            "error": None,
            "status": "running",
            "_async_task_id": "",
            "_task_queue_ref": None,
            "_llm_models": {},
        }
        result = await schedule_and_execute(state)
        assert len(result["ready_tasks"]) == 1
        assert result["ready_tasks"][0]["id"] == "b"

    @pytest.mark.asyncio
    async def test_merge_and_finish_success(self):
        """merge_and_finish 组装 final_output"""
        state: AgentGraphState = {
            "user_request": "test",
            "context": {},
            "plan": {"goal": "test goal", "reasoning": "test"},
            "plan_reasoning": "",
            "task_queue": [
                {"id": "a", "agent_type": "WritingAgent", "input": {"topic": "AI"}, "dependencies": []},
            ],
            "completed_task_ids": ["a"],
            "failed_task_ids": [],
            "task_results": {
                "a": {"agent_type": "WritingAgent", "output": {"content": "hello world"}, "status": "completed"},
            },
            "ready_tasks": [],
            "concurrency_limit": 3,
            "thinking_steps": [{"agent": "ManagerAgent", "action": "plan", "reasoning": "test", "status": "done"}],
            "trace_task_id": "test-5",
            "start_time": time.time(),
            "pending_decision": None,
            "decision_result": None,
            "final_output": None,
            "error": None,
            "status": "running",
            "_async_task_id": "",
            "_task_queue_ref": None,
            "_llm_models": {},
        }
        result = await merge_and_finish(state)
        assert result["status"] == "completed"
        assert result["final_output"] is not None
        assert "full_text" in result["final_output"]
        assert "hello world" in result["final_output"]["full_text"]

    @pytest.mark.asyncio
    async def test_merge_and_finish_partial(self):
        """部分任务失败 → status = partial"""
        state: AgentGraphState = {
            "user_request": "test",
            "context": {},
            "plan": None,
            "plan_reasoning": "",
            "task_queue": [
                {"id": "a", "agent_type": "WritingAgent", "input": {}, "dependencies": []},
                {"id": "b", "agent_type": "ReviewAgent", "input": {}, "dependencies": ["a"]},
            ],
            "completed_task_ids": ["a"],
            "failed_task_ids": ["b"],
            "task_results": {
                "a": {"agent_type": "WritingAgent", "output": {"content": "ok"}, "status": "completed"},
                "b": {"agent_type": "ReviewAgent", "output": {}, "status": "failed", "error": "bad"},
            },
            "ready_tasks": [],
            "concurrency_limit": 3,
            "thinking_steps": [],
            "trace_task_id": "test-6",
            "start_time": time.time(),
            "pending_decision": None,
            "decision_result": None,
            "final_output": None,
            "error": None,
            "status": "running",
            "_async_task_id": "",
            "_task_queue_ref": None,
            "_llm_models": {},
        }
        result = await merge_and_finish(state)
        assert result["status"] == "partial"


class TestShouldContinue:
    def test_should_continue_with_tasks(self):
        """有可执行的任务 → execute"""
        state: AgentGraphState = {
            "user_request": "test", "context": {},
            "plan": None, "plan_reasoning": "",
            "task_queue": [{"id": "a"}],
            "completed_task_ids": [], "failed_task_ids": [],
            "task_results": {}, "ready_tasks": [{"id": "a"}],
            "concurrency_limit": 3,
            "thinking_steps": [], "trace_task_id": "test", "start_time": time.time(),
            "pending_decision": None, "decision_result": None,
            "final_output": None, "error": None, "status": "running",
            "_async_task_id": "", "_task_queue_ref": None, "_llm_models": {},
        }
        assert should_continue_or_finish(state) == "execute"

    def test_should_continue_all_done(self):
        """所有任务完成 → done"""
        state: AgentGraphState = {
            "user_request": "test", "context": {},
            "plan": None, "plan_reasoning": "",
            "task_queue": [],
            "completed_task_ids": ["a", "b"],
            "failed_task_ids": [],
            "task_results": {
                "a": {}, "b": {},
            },
            "ready_tasks": [],
            "concurrency_limit": 3,
            "thinking_steps": [], "trace_task_id": "test", "start_time": time.time(),
            "pending_decision": None, "decision_result": None,
            "final_output": None, "error": None, "status": "running",
            "_async_task_id": "", "_task_queue_ref": None, "_llm_models": {},
        }
        assert should_continue_or_finish(state) == "done"


# ─────────────────────────────
# Orchestrator decomposed 测试（接口保留）
# ─────────────────────────────


class TestOrchestratorDecompose:
    @pytest.mark.asyncio
    async def test_decompose_simple_uses_llm(self, orchestrator):
        """简单任务现在也通过 LLM 拆解（之前跳过 LLM）"""
        plan = await orchestrator._decompose("write about AI", {"user": "test"})
        assert isinstance(plan, ExecutionPlan)
        # LLM path uses its own goal from the plan, not the truncated user request
        assert len(plan.tasks) == 2
        assert plan.tasks[0].agent_type == "WritingAgent"
        assert plan.tasks[1].agent_type == "ReviewAgent"
        assert plan.plan_method == "llm"

    @pytest.mark.asyncio
    async def test_decompose_medium_uses_llm(self, orchestrator):
        """中等任务通过 LLM 拆解"""
        plan = await orchestrator._decompose("写一篇优化小红书平台的技术文章", {"user": "test"})
        assert isinstance(plan, ExecutionPlan)
        assert plan.goal == "test goal"
        assert len(plan.tasks) == 2

    @pytest.mark.asyncio
    async def test_classify_complexity(self, orchestrator):
        assert orchestrator._classify_complexity("写一篇笔记") == "simple"
        assert orchestrator._classify_complexity("为一款智能笔记应用写产品推广文案") == "simple"
        assert orchestrator._classify_complexity("搜索AI趋势并优化小红书文章") == "medium"
        assert orchestrator._classify_complexity("生成带架构图和分析报告的技术博客") == "complex"
        assert orchestrator._classify_complexity("你好") == "simple"

    def test_parse_plan_from_json(self, orchestrator):
        response = json.dumps({
            "goal": "write article",
            "reasoning": "simple task",
            "tasks": [{"agent_type": "WritingAgent", "input": {"topic": "AI"}, "dependencies": []}],
        })
        plan = orchestrator._parse_plan(response, "write article")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].agent_type == "WritingAgent"

    def test_parse_plan_from_markdown(self, orchestrator):
        response = "```json\n" + json.dumps({
            "goal": "test", "reasoning": "test", "tasks": [],
        }) + "\n```"
        plan = orchestrator._parse_plan(response, "test")
        assert plan.goal == "test"

    @pytest.mark.asyncio
    async def test_parse_invalid_json_fallback(self, orchestrator):
        plan = orchestrator._parse_plan("invalid json", "fallback goal")
        assert len(plan.tasks) >= 1
        assert "parse error" in plan.reasoning


class TestOrchestratorFallback:
    @pytest.mark.asyncio
    async def test_fallback_llm_call(self, orchestrator):
        task = AgentTask(id="1", agent_type="UnknownAgent", input={"query": "hello"}, dependencies=[])
        result = await orchestrator._fallback_llm_call(task, {}, {})
        assert result.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_fallback_llm_call_error(self, mock_llm, orchestrator):
        mock_llm.generate_sync.side_effect = Exception("API error")
        orchestrator.llm_gateway = mock_llm
        task = AgentTask(id="1", agent_type="UnknownAgent", input={}, dependencies=[])
        result = await orchestrator._fallback_llm_call(task, {}, {})
        assert result.status == TaskStatus.FAILED
