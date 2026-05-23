"""反思工具 — 将 ReflectionEngine 方法暴露为 AgentTool

依赖：ReflectionEngine 实例（由 factory 创建时注入）
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ..base import AgentTool, AgentToolMeta

logger = logging.getLogger(__name__)


def create_reflection_tools(reflection_engine: Any) -> list[AgentTool]:
    """创建所有反思相关工具

    Args:
        reflection_engine: ReflectionEngine 实例（由 factory 传入）
    """

    async def _self_reflect_tool(
        agent_type: str,
        output_text: str,
        task_summary: str = "",
    ) -> str:
        """对 Agent 自身产出进行自省评分和改进建议

        Args:
            agent_type: 要自省的 Agent 类型
            output_text: 产出文本
            task_summary: 任务描述
        """
        if reflection_engine is None:
            return "错误：反思引擎未启用。"

        try:
            task_input = {"topic": task_summary} if task_summary else {}
            result = {"output": {"content": output_text}, "status": "completed"}

            ref_result = await reflection_engine.run_self_reflection(
                agent_type=agent_type,
                task_input=task_input,
                result=result,
                llm=None,
            )
            if ref_result is None:
                return "自省未执行（Agent 类型不在自省列表中或 LLM 不可用）。"

            lines = [
                f"自省结果（评分: {ref_result.score:.2f}）",
                f"通过: {'是' if ref_result.passed else '否'}",
            ]
            if ref_result.issues:
                lines.append(f"发现 {len(ref_result.issues)} 个问题:")
                for issue in ref_result.issues[:5]:
                    lines.append(f"  - {issue}")
            if ref_result.suggestions:
                lines.append("改进建议:")
                for s in ref_result.suggestions[:3]:
                    lines.append(f"  - {s}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning("self_reflect_tool 失败: %s", e)
            return f"自省执行失败: {e}"

    async def _request_peer_review(
        target_agent: str,
        output_text: str,
        task_summary: str = "",
    ) -> str:
        """请求其他 Agent 对指定产出进行同行评审

        Args:
            target_agent: 目标 Agent 类型
            output_text: 待评审的产出文本
            task_summary: 任务摘要
        """
        if reflection_engine is None:
            return "错误：反思引擎未启用。"

        try:
            target_output = {"content": output_text, "type": "text"}

            ref_result = await reflection_engine.run_peer_review(
                target_agent=target_agent,
                target_output=target_output,
                target_summary=task_summary,
                llm=None,
            )
            if ref_result is None:
                return "同行评审未执行（功能未启用或 LLM 不可用）。"

            lines = [
                f"同行评审结果（评分: {ref_result.score:.2f}）",
            ]
            if ref_result.issues:
                lines.append(f"问题 ({len(ref_result.issues)}):")
                for issue in ref_result.issues[:5]:
                    lines.append(f"  - {issue}")
            if ref_result.suggestions:
                lines.append("建议:")
                for s in ref_result.suggestions[:3]:
                    lines.append(f"  - {s}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning("request_peer_review 失败: %s", e)
            return f"同行评审失败: {e}"

    return [
        AgentTool(
            name="self_reflect",
            description="对自己的产出进行自省评分，发现问题和改进方向",
            execute_func=_self_reflect_tool,
            meta=AgentToolMeta(
                category="reflection",
                agent_types=["WritingAgent", "CodeAgent", "TranslationAgent", "SEOAgent"],
                builtin=True,
            ),
        ),
        AgentTool(
            name="request_peer_review",
            description="请求其他 Agent 对指定产出进行同行评审",
            execute_func=_request_peer_review,
            meta=AgentToolMeta(
                category="reflection",
                agent_types=["ManagerAgent", "WritingAgent"],
                builtin=True,
            ),
        ),
    ]
