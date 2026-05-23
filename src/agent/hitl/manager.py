"""HITLManager — 人机协同管理器

管理四种 HITL 门，包装现有 TaskQueue 的 decision_required/submit_decision 机制。
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

from .gates import (
    DecisionGate,
    ApprovalGate,
    CollaborativeEditGate,
    FeedbackGate,
)

logger = logging.getLogger(__name__)


class HITLManager:
    """人机协同管理器"""

    def __init__(self, task_queue: Any = None):
        self._task_queue = task_queue
        self._pending_gates: dict[str, Any] = {}      # gate_id -> gate
        self._gate_events: dict[str, asyncio.Event] = {}  # gate_id -> event
        self._gate_results: dict[str, Any] = {}        # gate_id -> result

    # ── 决策门 ──

    async def request_decision(
        self,
        gate: DecisionGate,
    ) -> Optional[str]:
        """提交决策门，等待人类选择"""
        self._pending_gates[gate.gate_id] = gate

        # 通过 TaskQueue 发布决策事件
        if self._task_queue:
            from ..workflow.task_queue import PendingDecision
            decision = PendingDecision(
                decision_id=gate.gate_id,
                task_id=gate.task_id,
                question=gate.question,
                options=gate.options,
                context=gate.context,
            )
            self._task_queue.set_decision_required(gate.task_id, decision)

        # 等待结果
        event = asyncio.Event()
        self._gate_events[gate.gate_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=gate.timeout)
            result = self._gate_results.pop(gate.gate_id, gate.default_value)
            gate.state = "resolved" if result else "timed_out"
            gate.result = result
            return result
        except asyncio.TimeoutError:
            gate.state = "timed_out"
            gate.result = gate.default_value
            logger.warning("HITL: decision %s timed out", gate.gate_id)
            return gate.default_value
        finally:
            self._gate_events.pop(gate.gate_id, None)

    # ── 审批门 ──

    async def request_approval(
        self,
        gate: ApprovalGate,
        timeout: float = 300.0,
    ) -> bool:
        """提交审批门，等待人类审批"""
        self._pending_gates[gate.gate_id] = gate

        if self._task_queue:
            from ..workflow.task_queue import PendingDecision
            decision = PendingDecision(
                decision_id=gate.gate_id,
                task_id=gate.task_id,
                question=f"审批 [{gate.target_agent}] 的产出：{gate.content_summary}",
                options=[
                    {"label": "通过", "value": "approved"},
                    {"label": "驳回", "value": "rejected"},
                ],
                context={"gate_type": "approval", "gate_id": gate.gate_id},
            )
            self._task_queue.set_decision_required(gate.task_id, decision)

        event = asyncio.Event()
        self._gate_events[gate.gate_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            result = self._gate_results.pop(gate.gate_id, "rejected")
            gate.state = "approved" if result == "approved" else "rejected"
            return result == "approved"
        except asyncio.TimeoutError:
            gate.state = "timed_out"
            return False
        finally:
            self._gate_events.pop(gate.gate_id, None)

    # ── 协作编辑门 ──

    async def request_collaborative_edit(
        self,
        gate: CollaborativeEditGate,
        timeout: float = 600.0,
    ) -> Optional[str]:
        """提交协作编辑门，等待人类编辑"""
        self._pending_gates[gate.gate_id] = gate

        if self._task_queue:
            from ..workflow.task_queue import PendingDecision
            decision = PendingDecision(
                decision_id=gate.gate_id,
                task_id=gate.task_id,
                question=f"编辑 [{gate.target_agent}] 的草稿",
                options=[
                    {"label": "已编辑完成", "value": "edited"},
                    {"label": "跳过编辑", "value": "skipped"},
                ],
                context={
                    "gate_type": "collaborative_edit",
                    "gate_id": gate.gate_id,
                    "draft": gate.draft_content[:2000],
                },
            )
            self._task_queue.set_decision_required(gate.task_id, decision)

        event = asyncio.Event()
        self._gate_events[gate.gate_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            result = self._gate_results.pop(gate.gate_id, None)
            if result:
                gate.edited_content = result.get("content", gate.draft_content)
                gate.state = "edited"
            else:
                gate.state = "skipped"
            return gate.edited_content
        except asyncio.TimeoutError:
            gate.state = "skipped"
            return gate.draft_content
        finally:
            self._gate_events.pop(gate.gate_id, None)

    # ── 反馈门 ──

    async def request_feedback(
        self,
        gate: FeedbackGate,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """提交反馈门，等待人类反馈"""
        self._pending_gates[gate.gate_id] = gate

        if self._task_queue:
            from ..workflow.task_queue import PendingDecision
            decision = PendingDecision(
                decision_id=gate.gate_id,
                task_id=gate.task_id,
                question=f"为 [{gate.target_agent}] 的内容提供反馈",
                options=[
                    {"label": "提交反馈", "value": "feedback"},
                    {"label": "无需反馈", "value": "skip"},
                ],
                context={
                    "gate_type": "feedback",
                    "gate_id": gate.gate_id,
                    "content": gate.content[:2000],
                },
            )
            self._task_queue.set_decision_required(gate.task_id, decision)

        event = asyncio.Event()
        self._gate_events[gate.gate_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            result = self._gate_results.pop(gate.gate_id, None)
            gate.feedback = result if isinstance(result, str) else None
            gate.state = "received" if gate.feedback else "acknowledged"
            return gate.feedback
        except asyncio.TimeoutError:
            gate.state = "acknowledged"
            return None
        finally:
            self._gate_events.pop(gate.gate_id, None)

    # ── 结果提交（API 层调用）──

    def submit_decision_result(
        self,
        gate_id: str,
        result: Any,
    ) -> bool:
        """人类提交决策结果"""
        if gate_id not in self._pending_gates:
            logger.warning("HITL: unknown gate %s", gate_id)
            return False

        self._gate_results[gate_id] = result
        if gate_id in self._gate_events:
            self._gate_events[gate_id].set()

        # 同时更新 TaskQueue
        gate = self._pending_gates[gate_id]
        if self._task_queue and hasattr(gate, "task_id"):
            self._task_queue.submit_decision(
                gate.task_id,
                gate_id,
                str(result),
            )

        return True

    # ── 查询 ──

    def get_pending_gates(self) -> list[dict]:
        """获取所有待处理的门"""
        return [
            g.to_dict() if hasattr(g, "to_dict") else {"gate_id": g_id}
            for g_id, g in self._pending_gates.items()
            if getattr(g, "state", "") == "pending"
        ]

    def get_gate(self, gate_id: str) -> Optional[Any]:
        return self._pending_gates.get(gate_id)

    def has_pending(self) -> bool:
        return any(
            getattr(g, "state", "") == "pending"
            for g in self._pending_gates.values()
        )

    def clear(self) -> None:
        """清空状态"""
        self._pending_gates.clear()
        self._gate_events.clear()
        self._gate_results.clear()
