"""HITL — 人机协同门与管理器单元测试"""
import asyncio
import pytest
import uuid

from src.agent.hitl import HITLManager
from src.agent.hitl.gates import (
    DecisionGate, ApprovalGate, CollaborativeEditGate, FeedbackGate,
)


class TestDecisionGate:
    """决策门测试"""

    def test_create_decision_gate(self):
        gate = DecisionGate(
            gate_id="d1",
            task_id="task1",
            question="选择发布方案？",
            options=[
                {"label": "方案A", "value": "plan_a"},
                {"label": "方案B", "value": "plan_b"},
            ],
        )
        assert gate.gate_id == "d1"
        assert gate.state == "pending"
        assert gate.result is None
        assert gate.default_value is None

    def test_to_dict(self):
        gate = DecisionGate(
            gate_id="d1", task_id="t1",
            question="Q?", options=[{"label": "是", "value": "yes"}],
        )
        d = gate.to_dict()
        assert d["gate_id"] == "d1"
        assert d["state"] == "pending"

    def test_from_dict(self):
        data = {
            "gate_id": "d2", "task_id": "t2",
            "question": "继续？", "options": [{"label": "继续", "value": "go"}],
            "state": "pending",
        }
        gate = DecisionGate.from_dict(data)
        assert gate.gate_id == "d2"
        assert gate.question == "继续？"

    def test_timeout_default(self):
        gate = DecisionGate(
            gate_id="d1", task_id="t1",
            question="Q?", options=[],
            timeout=300.0, default_value=None,
        )
        assert gate.timeout == 300.0

    def test_custom_timeout(self):
        gate = DecisionGate(
            gate_id="d1", task_id="t1",
            question="Q?", options=[],
            timeout=60.0, default_value="no",
        )
        assert gate.timeout == 60.0
        assert gate.default_value == "no"

    def test_resolved_state(self):
        gate = DecisionGate(
            gate_id="d1", task_id="t1",
            question="Q?", options=[{"label": "是", "value": "yes"}],
        )
        gate.state = "resolved"
        gate.result = "yes"
        assert gate.state == "resolved"
        assert gate.result == "yes"


class TestApprovalGate:
    """审批门测试"""

    def test_create_approval_gate(self):
        gate = ApprovalGate(
            gate_id="a1",
            task_id="task1",
            target_agent="WritingAgent",
            content_summary="文章草稿摘要",
            full_content="全文内容...",
            instructions="检查事实准确性",
        )
        assert gate.state == "pending"
        assert gate.feedback == ""

    def test_to_dict(self):
        gate = ApprovalGate(
            gate_id="a1", task_id="t1",
            target_agent="Writer", content_summary="summary",
        )
        d = gate.to_dict()
        assert d["gate_type"] == "approval"
        assert d["target_agent"] == "Writer"

    def test_approve(self):
        gate = ApprovalGate(
            gate_id="a1", task_id="t1",
            target_agent="Writer", content_summary="s",
        )
        gate.state = "approved"
        assert gate.state == "approved"

    def test_reject_with_feedback(self):
        gate = ApprovalGate(
            gate_id="a1", task_id="t1",
            target_agent="Writer", content_summary="s",
        )
        gate.state = "rejected"
        gate.feedback = "需要更多数据支持"
        assert gate.feedback == "需要更多数据支持"


class TestCollaborativeEditGate:
    """协作编辑门测试"""

    def test_create_edit_gate(self):
        gate = CollaborativeEditGate(
            gate_id="e1",
            task_id="task1",
            target_agent="WritingAgent",
            draft_content="初始草稿内容...",
            edit_instructions="请优化开头段落",
        )
        assert gate.state == "pending"
        assert gate.edited_content is None
        assert gate.iteration_count == 0
        assert gate.max_iterations == 3

    def test_edit_content(self):
        gate = CollaborativeEditGate(
            gate_id="e1", task_id="t1",
            target_agent="Writer", draft_content="original",
        )
        gate.edited_content = "edited version"
        gate.state = "edited"
        assert gate.edited_content == "edited version"
        assert gate.state == "edited"

    def test_to_dict(self):
        gate = CollaborativeEditGate(
            gate_id="e1", task_id="t1",
            target_agent="Writer", draft_content="draft",
        )
        d = gate.to_dict()
        assert d["gate_type"] == "collaborative_edit"


class TestFeedbackGate:
    """反馈门测试"""

    def test_create_feedback_gate(self):
        gate = FeedbackGate(
            gate_id="f1",
            task_id="task1",
            target_agent="WritingAgent",
            content="待反馈内容",
        )
        assert gate.state == "pending"
        assert gate.feedback is None

    def test_receive_feedback(self):
        gate = FeedbackGate(
            gate_id="f1", task_id="t1",
            target_agent="Writer", content="content",
        )
        gate.feedback = "写得很好，但需要更多例子"
        gate.state = "received"
        assert gate.feedback == "写得很好，但需要更多例子"

    def test_to_dict(self):
        gate = FeedbackGate(
            gate_id="f1", task_id="t1",
            target_agent="Writer", content="content",
        )
        d = gate.to_dict()
        assert d["gate_type"] == "feedback"


class TestHITLManager:
    """HITLManager 管理器测试"""

    @pytest.fixture
    def manager(self):
        return HITLManager()

    def test_no_pending_initially(self, manager):
        assert manager.has_pending() is False
        assert manager.get_pending_gates() == []

    def test_get_nonexistent_gate(self, manager):
        assert manager.get_gate("nonexistent") is None

    @pytest.mark.asyncio
    async def test_request_decision_timeout(self, manager):
        """决策门超时时返回默认值"""
        gate = DecisionGate(
            gate_id="d1", task_id="t1",
            question="继续？", options=[{"label": "是", "value": "yes"}],
            timeout=0.1, default_value="no",
        )
        result = await manager.request_decision(gate)
        assert result == "no"
        assert gate.state == "timed_out"

    @pytest.mark.asyncio
    async def test_request_decision_resolved(self, manager):
        gate = DecisionGate(
            gate_id="d2", task_id="t1",
            question="选择？", options=[{"label": "A", "value": "a"}],
            timeout=5.0,
        )

        import asyncio
        async def resolve_later():
            await asyncio.sleep(0.05)
            manager.submit_decision_result("d2", "a")

        asyncio.create_task(resolve_later())
        result = await manager.request_decision(gate)
        assert result == "a"
        assert gate.state == "resolved"

    @pytest.mark.asyncio
    async def test_request_approval_timeout(self, manager):
        gate = ApprovalGate(
            gate_id="a1", task_id="t1",
            target_agent="Writer", content_summary="summary",
        )
        result = await manager.request_approval(gate, timeout=0.1)
        assert result is False  # timeout defaults to rejected
        assert gate.state == "timed_out"

    @pytest.mark.asyncio
    async def test_request_approval_approved(self, manager):
        gate = ApprovalGate(
            gate_id="a2", task_id="t1",
            target_agent="Writer", content_summary="s",
        )

        import asyncio
        async def approve_later():
            await asyncio.sleep(0.05)
            manager.submit_decision_result("a2", "approved")

        asyncio.create_task(approve_later())
        result = await manager.request_approval(gate, timeout=5.0)
        assert result is True
        assert gate.state == "approved"

    @pytest.mark.asyncio
    async def test_request_feedback_timeout(self, manager):
        gate = FeedbackGate(
            gate_id="f1", task_id="t1",
            target_agent="Writer", content="content",
        )
        result = await manager.request_feedback(gate, timeout=0.1)
        assert result is None  # timeout returns None
        assert gate.state == "acknowledged"

    @pytest.mark.asyncio
    async def test_request_feedback_received(self, manager):
        gate = FeedbackGate(
            gate_id="f2", task_id="t1",
            target_agent="Writer", content="content",
        )

        import asyncio
        async def feedback_later():
            await asyncio.sleep(0.05)
            manager.submit_decision_result("f2", "写得好！")

        asyncio.create_task(feedback_later())
        result = await manager.request_feedback(gate, timeout=5.0)
        assert result == "写得好！"
        assert gate.state == "received"

    @pytest.mark.asyncio
    async def test_collaborative_edit_timeout(self, manager):
        gate = CollaborativeEditGate(
            gate_id="e1", task_id="t1",
            target_agent="Writer", draft_content="draft",
        )
        result = await manager.request_collaborative_edit(gate, timeout=0.1)
        assert result == "draft"  # timeout returns original draft
        assert gate.state == "skipped"

    @pytest.mark.asyncio
    async def test_submit_decision_unknown_gate(self, manager):
        result = manager.submit_decision_result("nonexistent", "yes")
        assert result is False

    def test_has_pending_after_request(self, manager):
        """pending gate 应被记录"""
        import asyncio
        gate = DecisionGate(
            gate_id="dp1", task_id="t1",
            question="?", options=[{"label": "是", "value": "yes"}],
            timeout=60.0,
        )
        manager._pending_gates[gate.gate_id] = gate
        assert manager.has_pending() is True
        assert len(manager.get_pending_gates()) == 1

    def test_get_pending_gates_returns_dicts(self, manager):
        gate = DecisionGate(
            gate_id="dp2", task_id="t1",
            question="test?", options=[],
        )
        manager._pending_gates[gate.gate_id] = gate
        gates = manager.get_pending_gates()
        assert isinstance(gates[0], dict)
        assert gates[0]["gate_id"] == "dp2"

    def test_clear_removes_all_state(self, manager):
        gate = DecisionGate(
            gate_id="d1", task_id="t1",
            question="?", options=[],
        )
        manager._pending_gates[gate.gate_id] = gate
        manager._gate_events["d1"] = asyncio.Event()
        manager._gate_results["d1"] = "yes"

        manager.clear()
        assert manager._pending_gates == {}
        assert manager._gate_events == {}
        assert manager._gate_results == {}
