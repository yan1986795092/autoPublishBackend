"""同行评审（Peer Review）：其他 Agent 评审产出"""
from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .models import ReflectionResult
from src.prompts import PromptBuilder

logger = logging.getLogger(__name__)


async def peer_review(
    reviewer_agent: str,
    target_agent: str,
    target_output: dict,
    target_task_summary: str,
    llm: Optional[ChatOpenAI],
) -> Optional[ReflectionResult]:
    """同行评审：reviewer_agent 评审 target_agent 的产出"""
    content = (target_output or {}).get("content", "")
    if not content or len(content.strip()) < 100:
        return None

    if not llm:
        return None

    try:
        built = (PromptBuilder()
            .system("system.reviewer")
            .instructions("reflection.peer", {
                "reviewer_agent": reviewer_agent,
                "target_agent": target_agent,
                "task_summary": target_task_summary,
                "output_content": content[:2500],
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

        return _parse_peer_review(reviewer_agent, raw)
    except Exception as e:
        logger.warning("同行评审失败: %s", e)
        return None


def _parse_peer_review(
    reviewer_agent: str,
    raw: str,
) -> Optional[ReflectionResult]:
    """解析同行评审结果"""
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].strip()
        data = json.loads(raw)

        score = data.get("overall_score", 0.7)
        if isinstance(score, (int, float)):
            score = float(score)
            if score > 1.0:
                score /= 5.0  # 5 分制转 0-1
        else:
            score = 0.7

        issues = data.get("issues", [])
        suggestions = data.get("suggestions", [])

        return ReflectionResult(
            agent_type=reviewer_agent,
            target_task_id="",
            reflection_type="peer",
            score=round(min(score, 1.0), 2),
            issues=issues,
            suggestions=suggestions,
            passed=score >= 0.7 and len(issues) == 0,
            metadata={"target_agent": raw[:50]},
        )
    except (json.JSONDecodeError, ValueError):
        logger.warning("解析同行评审失败: JSON 解析错误")
        return None
