"""HITL 工具 — 将 HITLManager 方法暴露为 AgentTool

依赖：HITLManager 实例（由 factory 创建时注入）
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from ..base import AgentTool, AgentToolMeta

logger = logging.getLogger(__name__)


def create_hitl_tools(hitl_manager: Any) -> list[AgentTool]:
    """创建所有人机协同相关工具

    Args:
        hitl_manager: HITLManager 实例（由 factory 传入）
    """

    async def _request_decision(
        question: str,
        options_json: str = "",
        context_json: str = "",
    ) -> str:
        """请求人类做出决策选择

        Args:
            question: 需要人类决策的问题
            options_json: 选项的 JSON 字符串（如 [{"label":"通过","value":"approve"}]）
            context_json: 上下文信息的 JSON 字符串
        """
        if hitl_manager is None:
            return "错误：HITL 未启用。"

        try:
            from ..hitl.gates import DecisionGate

            options = json.loads(options_json) if options_json else [
                {"label": "是", "value": "yes"},
                {"label": "否", "value": "no"},
            ]
            context = json.loads(context_json) if context_json else {}

            gate = DecisionGate(
                gate_id=str(uuid.uuid4()),
                task_id="",
                question=question,
                options=options,
                context=context,
            )
            result = await hitl_manager.request_decision(gate)
            return f"决策结果: {result}"
        except Exception as e:
            logger.warning("request_decision 工具失败: %s", e)
            return f"决策请求失败: {e}"

    async def _request_approval(
        target_agent: str,
        content_summary: str,
    ) -> str:
        """请求人类审批指定 Agent 的产出

        Args:
            target_agent: 需要审批的 Agent 类型
            content_summary: 产出摘要
        """
        if hitl_manager is None:
            return "错误：HITL 未启用。"

        try:
            from ..hitl.gates import ApprovalGate

            gate = ApprovalGate(
                gate_id=str(uuid.uuid4()),
                task_id="",
                target_agent=target_agent,
                content_summary=content_summary,
            )
            approved = await hitl_manager.request_approval(gate)
            return "审批通过。继续执行。" if approved else "审批被驳回。"
        except Exception as e:
            logger.warning("request_approval 工具失败: %s", e)
            return f"审批请求失败: {e}"

    async def _request_feedback(
        target_agent: str,
        content: str,
    ) -> str:
        """请求人类对指定内容提供反馈

        Args:
            target_agent: 目标 Agent 类型
            content: 待反馈的内容
        """
        if hitl_manager is None:
            return "错误：HITL 未启用。"

        try:
            from ..hitl.gates import FeedbackGate

            gate = FeedbackGate(
                gate_id=str(uuid.uuid4()),
                task_id="",
                target_agent=target_agent,
                content=content,
            )
            feedback = await hitl_manager.request_feedback(gate)
            if feedback:
                return f"收到反馈: {feedback[:500]}"
            return "人类未提供反馈（超时或跳过）。"
        except Exception as e:
            logger.warning("request_feedback 工具失败: %s", e)
            return f"反馈请求失败: {e}"

    async def _check_pending_gates() -> str:
        """检查是否有待处理的人机协同请求"""
        if hitl_manager is None:
            return "错误：HITL 未启用。"

        try:
            gates = hitl_manager.get_pending_gates()
            if not gates:
                return "无待处理的 HITL 请求。"
            lines = [f"有 {len(gates)} 个待处理的 HITL 请求："]
            for g in gates:
                lines.append(f"  - {g.get('gate_id', '?')}: {g.get('question', '')[:100]}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning("check_pending_gates 工具失败: %s", e)
            return f"检查 HITL 请求失败: {e}"

    return [
        AgentTool(
            name="request_decision",
            description="向人类请求决策（如选择方案、确认方向），将暂停等待人类响应",
            execute_func=_request_decision,
            meta=AgentToolMeta(
                category="hitl",
                agent_types=["ManagerAgent"],
                builtin=True,
            ),
        ),
        AgentTool(
            name="request_approval",
            description="请求人类审批 Agent 的产出结果",
            execute_func=_request_approval,
            meta=AgentToolMeta(
                category="hitl",
                agent_types=["ManagerAgent", "ReviewAgent"],
                builtin=True,
            ),
        ),
        AgentTool(
            name="request_feedback",
            description="请求人类对内容提供反馈意见",
            execute_func=_request_feedback,
            meta=AgentToolMeta(
                category="hitl",
                agent_types=["ManagerAgent", "WritingAgent"],
                builtin=True,
            ),
        ),
        AgentTool(
            name="check_pending_gates",
            description="检查是否有待处理的人机协同请求",
            execute_func=_check_pending_gates,
            meta=AgentToolMeta(
                category="hitl",
                agent_types=["ManagerAgent"],
                builtin=True,
            ),
        ),
    ]
