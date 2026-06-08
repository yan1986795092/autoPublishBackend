"""重试与容错机制

提供：
- run_agent_with_retry: 带重试和质量检查的 Agent 执行
- QualityError: 质量不达标异常
- QUALITY_THRESHOLD: 质量阈值常量
"""

from __future__ import annotations
import asyncio
from typing import Any, Awaitable, Callable, Optional, TypeVar

from .schemas import AgentResult, TaskStatus, UsageInfo
from src.logging_shared.config import get_logger

logger = get_logger("agent.retry")

T = TypeVar("T")


class QualityError(Exception):
    """Agent 输出质量不达标"""
    def __init__(self, message: str, score: float = 0.0, details: Optional[dict] = None):
        super().__init__(message)
        self.score = score
        self.details = details or {}


QUALITY_THRESHOLD = 60.0  # 百分制


def simple_quality_score(output: dict) -> float:
    """对 Agent 输出做简单的质量评分

    基于内容长度、丰富度、结构等指标，返回 0-100 的分数。
    适用于无 LLM 场景的降级评估。真实质量判断由 Reflection 机制负责。
    """
    content = output.get("content", "")
    if not content:
        return 0.0

    score = 50.0  # base

    # 长度检查（核心指标）：按区间加分，上限封顶
    length = len(content)
    if length < 50:
        return max(0.0, score - 20)
    elif length < 200:
        score += 10
    elif length < 1000:
        score += 20
    elif length < 3000:
        score += 30
    elif length < 8000:
        score += 40
    else:
        score += 40

    # 结构完整性：段落数量加分
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(paragraphs) >= 3:
        score += 10
    elif len(paragraphs) >= 1:
        score += 5

    # Markdown 结构：标题加分
    heading_count = content.count("\n##") + content.count("\n#")
    if heading_count >= 2:
        score += 5

    # 代码块：统计完整对（``` 开 + ``` 闭），奇数对 = 有未闭合块
    triple_backtick_count = content.count("```")
    complete_pairs = triple_backtick_count // 2
    if complete_pairs >= 2:
        score += 5
    if triple_backtick_count % 2 != 0:
        score -= 15  # 未闭合的代码块（影响可读性）

    # 扣分项：占位符检测
    import re
    placeholder_patterns = [
        (re.compile(r'\bTODO\b', re.I), "TODO keyword"),
        ("待补充", "待补充"),
        ("这里应该", "这里应该"),
        ("placeholder", "placeholder"),
        ("xxx", "xxx"),
        ("TBD", "TBD"),
        ("此处应填", "此处应填"),
        ("稍后补充", "稍后补充"),
        ("内容待定", "内容待定"),
        ("【】", "空括号标记"),
        ("「」", "空引号标记"),
        ("_____", "下划线占位"),
    ]
    for item in placeholder_patterns:
        if isinstance(item, tuple):
            pattern, _ = item
        else:
            pattern = item
        if isinstance(pattern, re.Pattern):
            if pattern.search(content):
                score -= 10
                break
        elif isinstance(pattern, str):
            if pattern.lower() in content.lower():
                score -= 10
                break

    # 过度重复检测：唯一行占比低于阈值说明内容大量重复
    lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
    if len(lines) > 10:
        unique_lines = len(set(lines))
        if unique_lines / len(lines) < 0.5:
            score -= 20

    return max(0.0, min(100.0, score))


async def run_agent_with_retry(
    agent_func: Callable[..., Awaitable[dict]],
    task_input: dict,
    llm: Any,
    state: dict,
    max_retries: int = 3,
    quality_threshold: float = QUALITY_THRESHOLD,
    agent_type: str = "",
    task_label: str = "",
    context: Optional[dict] = None,
) -> dict:
    """执行 Agent 函数，带重试和质量检查

    Args:
        agent_func: Agent 处理函数
        task_input: 输入参数
        llm: LLM 实例
        state: 状态
        max_retries: 最大重试次数
        quality_threshold: 质量阈值（0-100）
        agent_type: Agent 类型名称
        task_label: 任务描述（日志用）
        context: 额外的上下文（如 feedback 等）

    Returns:
        Agent 执行结果 dict
    """
    if context is None:
        context = {}

    last_error = ""
    feedback_history: list[str] = []

    for attempt in range(max_retries):
        try:
            # 指数退避（第一次不等待）
            if attempt > 0:
                wait = min(2 ** attempt, 30)
                logger.info("重试中", agent_type=agent_type, attempt=attempt + 1, max_retries=max_retries, wait_s=wait)
                await asyncio.sleep(wait)

            # 构建每次重试的输入（不修改原始 task_input）
            retry_input = dict(task_input)
            if feedback_history:
                retry_input["_feedback"] = feedback_history[-1]

            result = await agent_func(retry_input, llm, state)
            output = result.get("output", {})

            # 检查是否失败
            if result.get("status") == "failed":
                last_error = result.get("error", "Unknown error")
                logger.warning("Agent 执行失败", agent_type=agent_type, attempt=attempt + 1, error=last_error)
                feedback_history.append(f"上一步执行失败，请修正: {last_error}")
                continue

            # 质量评分
            score = simple_quality_score(output)
            result["_quality_score"] = score

            if score < quality_threshold:
                logger.info("质量评分不足", agent_type=agent_type, attempt=attempt + 1, score=score, threshold=quality_threshold)
                feedback_history.append(f"质量评分 {score:.1f} 不足，请更详细地展开论述，增加深度和细节")
                last_error = f"质量评分 {score:.1f} 低于阈值"
                if "thinking_steps" in result:
                    for step in result["thinking_steps"]:
                        step["status"] = "retry"
                continue

            logger.info("通过质量检查", agent_type=agent_type, score=score)
            return result

        except (asyncio.TimeoutError, ConnectionError, OSError) as e:
            last_error = str(e)
            logger.warning("网络/超时错误", agent_type=agent_type, attempt=attempt + 1, error=str(e))
            feedback_history.append(f"网络/超时错误: {last_error}, 请重试")
            await asyncio.sleep(2 ** attempt)

        except QualityError as e:
            last_error = str(e)
            feedback_history.append(f"质量问题: {e}")
            logger.warning("质量问题", agent_type=agent_type, attempt=attempt + 1, error=str(e))

        except Exception as e:
            last_error = str(e)
            logger.error("意外错误", agent_type=agent_type, attempt=attempt + 1, error=str(e))
            if attempt == max_retries - 1:
                return {
                    "status": "failed",
                    "agent_type": agent_type,
                    "error": f"所有重试失败: {last_error}",
                    "output": {"content": "", "fallback": True},
                    "tokens_used": 0,
                    "thinking_steps": [{
                        "agent": agent_type,
                        "action": "retry_exhausted",
                        "reasoning": f"重试 {max_retries} 次均失败",
                        "output_summary": f"Error: {last_error}",
                        "tokens_used": 0,
                        "status": "failed",
                    }],
                }

    # 所有重试耗尽，返回降级结果
    logger.warning("重试耗尽，返回降级结果", agent_type=agent_type, max_retries=max_retries)
    return {
        "status": "partial",
        "agent_type": agent_type,
        "output": {
            "content": f"（{agent_type} 经过 {max_retries} 次重试后仍未能达到质量要求）",
            "fallback": True,
            "error": last_error,
        },
        "tokens_used": 0,
        "thinking_steps": [{
            "agent": agent_type,
            "action": "quality_fallback",
            "reasoning": f"质量评分不足，跳过该步骤",
            "output_summary": f"Fallback after {max_retries} retries",
            "tokens_used": 0,
            "status": "failed",
        }],
    }
