"""元认知（Meta-Cognition）：ManagerAgent 整体评审"""
from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .models import ReflectionResult
from src.prompts import PromptBuilder

logger = logging.getLogger(__name__)


async def meta_reflect(
    plan: dict,
    task_results: dict,
    completed_ids: list[str],
    llm: Optional[ChatOpenAI],
) -> Optional[ReflectionResult]:
    """元认知评审：ManagerAgent 视角评估整体执行质量"""
    completed_count = len(completed_ids)
    total_count = len(task_results)

    if total_count == 0:
        return None

    # 收集各任务摘要
    summaries = []
    for task_id, result in task_results.items():
        output = result.get("output", {}) or {}
        content = (output or {}).get("content", "")
        summaries.append(
            f"  [{result.get('agent_type', '?')}] "
            f"status={result.get('status', '?')}, "
            f"len={len(content or '')} chars"
        )

    goal = (plan or {}).get("goal", "")[:100]

    meta_prompt = f"""作为内容生产经理，评审本次整体执行质量。

目标：{goal}

执行概况：
- 完成: {completed_count}/{total_count} 任务
- 各任务摘要：
{chr(10).join(summaries)}

请评估：
1. 目标是否达成
2. 各任务衔接是否顺畅
3. 整体产出质量
4. 改进建议

以 JSON 格式输出：
{{
  "goal_achieved": true/false,
  "overall_score": 0.0-1.0,
  "issues": [str],
  "suggestions": [str],
  "summary": "一句话评估"
}}"""

    if not llm:
        return _rule_based_meta(goal, completed_count, total_count)

    try:
        response = await llm.ainvoke([
            SystemMessage(content="你是内容生产经理，负责整体质量把控。"),
            HumanMessage(content=meta_prompt),
        ])
        raw = response.content.strip()
        return _parse_meta_result(raw)
    except Exception as e:
        logger.warning("meta_reflect LLM 失败: %s", e)
        return _rule_based_meta(goal, completed_count, total_count)


def _parse_meta_result(raw: str) -> Optional[ReflectionResult]:
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].strip()
        data = json.loads(raw)

        score = data.get("overall_score", 0.7)
        if isinstance(score, (int, float)):
            score = float(min(score, 1.0))
        else:
            score = 0.7

        return ReflectionResult(
            agent_type="ManagerAgent",
            target_task_id="",
            reflection_type="meta",
            score=score,
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            passed=data.get("goal_achieved", score >= 0.7),
            metadata={"summary": data.get("summary", "")},
        )
    except (json.JSONDecodeError, ValueError):
        return None


def _rule_based_meta(
    goal: str,
    completed: int,
    total: int,
) -> ReflectionResult:
    """基于规则的元认知兜底"""
    ratio = completed / max(total, 1)
    issues = []
    if ratio < 1.0:
        issues.append(f"有 {total - completed} 个任务未完成")

    return ReflectionResult(
        agent_type="ManagerAgent",
        target_task_id="",
        reflection_type="meta",
        score=ratio,
        issues=issues,
        passed=ratio >= 0.7,
        metadata={"summary": f"完成 {completed}/{total} 任务"},
    )
