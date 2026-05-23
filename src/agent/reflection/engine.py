"""ReflectionEngine — 反思编排引擎

管理三层反思的执行流程：
  自省 → 同行评审 → 元认知
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from .models import ReflectionResult, ReflectionConfig
from .self_reflection import self_reflect
from .peer_review import peer_review
from .meta_cognition import meta_reflect

logger = logging.getLogger(__name__)


class ReflectionEngine:
    """反思编排引擎"""

    def __init__(
        self,
        config: Optional[ReflectionConfig] = None,
        registry: Any = None,  # AgentRegistry
    ):
        self.config = config or ReflectionConfig()
        self.registry = registry

    # ── 自省 ──

    async def run_self_reflection(
        self,
        agent_type: str,
        task_input: dict,
        result: dict,
        llm: Optional[ChatOpenAI],
    ) -> Optional[ReflectionResult]:
        """执行自省"""
        if agent_type not in self.config.self_reflection_agents:
            return None
        return await self_reflect(agent_type, task_input, result, llm)

    # ── 同行评审 ──

    async def run_peer_review(
        self,
        target_agent: str,
        target_output: dict,
        target_summary: str,
        llm: Optional[ChatOpenAI],
    ) -> Optional[ReflectionResult]:
        """执行同行评审"""
        if not self.config.enable_peer_review:
            return None

        # 从配置中获取审阅者
        for reviewer in self.config.peer_reviewers:
            if reviewer == target_agent:
                continue
            return await peer_review(
                reviewer_agent=reviewer,
                target_agent=target_agent,
                target_output=target_output,
                target_task_summary=target_summary,
                llm=llm,
            )
        return None

    # ── 元认知 ──

    async def run_meta_reflection(
        self,
        plan: dict,
        task_results: dict,
        completed_ids: list[str],
        llm: Optional[ChatOpenAI],
    ) -> Optional[ReflectionResult]:
        """执行元认知评审"""
        if not self.config.enable_meta_cognition:
            return None
        return await meta_reflect(plan, task_results, completed_ids, llm)

    # ── 决策 ──

    def should_retry(self, result: ReflectionResult, round_num: int) -> bool:
        """判断是否需要重试"""
        if round_num >= self.config.max_self_reflection_rounds:
            return False
        if result.metadata.get("skip"):
            return False
        return result.score < self.config.quality_threshold or not result.passed

    def should_continue(self, results: list[ReflectionResult]) -> bool:
        """判断整体是否应继续执行"""
        if not results:
            return True
        return any(r.passed for r in results)
