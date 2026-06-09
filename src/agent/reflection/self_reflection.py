"""自省（Self-Reflection）：Agent 对自己产出的质量评判"""
from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .models import ReflectionResult
from src.prompts import PromptBuilder

logger = logging.getLogger(__name__)


async def self_reflect(
    agent_type: str,
    task_input: dict,
    result: dict,
    llm: Optional[ChatOpenAI],
) -> Optional[ReflectionResult]:
    """Agent 对自己的产出进行自省

    使用专门的反思 prompt 让 LLM 评分自己的输出。
    如果 LLM 不可用，返回基于规则的简单评分。
    """
    output = result.get("output", {}) or {}
    content = (output or {}).get("content", "")
    if not content or len(content.strip()) < 50:
        return ReflectionResult(
            agent_type=agent_type,
            target_task_id=task_input.get("_task_id", ""),
            reflection_type="self",
            score=1.0,
            passed=True,
            issues=[],
            suggestions=[],
            metadata={"reason": "跳过: 内容过短"},
        )

    # LLM 自省
    if llm:
        try:
            # 构建反思 prompt
            topic = task_input.get("topic", task_input.get("query", ""))[:100]
            built = (PromptBuilder()
                .system("system.reviewer")
                .instructions("reflection.self", {
                    "agent_type": agent_type,
                    "task_summary": topic,
                    "output_content": content[:2000],
                })
                .build())

            lc_msgs = []
            for m in built.messages:
                if m["role"] == "system":
                    lc_msgs.append(SystemMessage(content=m["content"]))
                else:
                    lc_msgs.append(HumanMessage(content=m["content"]))

            response = await llm.ainvoke(lc_msgs)
            raw = response.content.strip()

            return _parse_reflection(agent_type, task_input, raw)
        except Exception as e:
            logger.warning("self_reflect LLM 失败: %s", e)

    # 规则兜底
    return _rule_based_reflection(agent_type, task_input, content)


def _parse_reflection(
    agent_type: str,
    task_input: dict,
    raw: str,
) -> ReflectionResult:
    """解析 LLM 返回的反思 JSON"""
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].strip()
        data = json.loads(raw)

        scores = data.get("scores", {})
        score_list = [v for v in scores.values() if isinstance(v, (int, float))]
        avg_score = sum(score_list) / len(score_list) / 5.0 if score_list else 0.7

        issues = data.get("issues", [])
        suggestions = data.get("suggestions", [])

        return ReflectionResult(
            agent_type=agent_type,
            target_task_id=task_input.get("_task_id", ""),
            reflection_type="self",
            score=round(avg_score, 2),
            issues=issues,
            suggestions=suggestions,
            passed=avg_score >= 0.7 and len(issues) == 0,
        )
    except (json.JSONDecodeError, ValueError, ZeroDivisionError) as e:
        logger.warning("解析反思失败: %s", e)
        return _rule_based_reflection(agent_type, task_input, raw)


def _rule_based_reflection(
    agent_type: str,
    task_input: dict,
    content: str,
) -> ReflectionResult:
    """基于规则的简单自省兜底"""
    issues = []

    # 长度检查
    if len(content) < 100:
        issues.append("内容过短")

    # 重复检查
    sentences = set()
    for line in content.split("。"):
        stripped = line.strip()[:20]
        if stripped in sentences:
            issues.append("存在重复内容")
            break
        sentences.add(stripped)

    # 通用指标
    has_title = any(marker in content[:100] for marker in ["# ", "## ", "标题："])
    if agent_type == "WritingAgent" and not has_title:
        issues.append("缺少标题")

    passed = len(issues) == 0
    score = 1.0 if passed else 0.5

    return ReflectionResult(
        agent_type=agent_type,
        target_task_id=task_input.get("_task_id", ""),
        reflection_type="self",
        score=score,
        issues=issues,
        passed=passed,
    )
