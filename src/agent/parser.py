"""Structured-output parser with reask loop (Layer 2).

Pydantic 解析 + 拓扑校验双重验证，失败时把具体问题注入下轮 LLM 重新生成。
不自动修复任何问题——所有问题都交给 LLM 在下一轮纠正。

为什么不用自动修复：
- 自动加的 SearchAnalyzer / FixAgent input 是空模板，质量差
- 复杂问题（循环依赖、缺失链路）只有 LLM 能正确处理
- 一次额外 LLM 调用换来 plan 质量大幅提升
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from pydantic import ValidationError
from langchain_core.messages import SystemMessage, HumanMessage

from .contracts import PlannerOutput, DEFAULT_RELIABILITY_CONFIG
from .plan_validator import validate_plan
from ..logging_shared import get_logger

logger = get_logger("agent.parser")


REASK_SYSTEM_PROMPT = """你之前的回复存在以下问题，请修正后重新输出：

{issues}

请严格按以下 schema 输出（只输出 JSON，不要任何解释/前缀/后缀）：

```json
{schema}
```

可用 Agent：
{available_agents}

规则：
- agent_type 必须从上方"可用 Agent"列表中选择
- 每个 task 必须有唯一 id；dependencies 中的 id 必须引用本 plan 中其他 task 的 id
- 必须有至少一个根任务（dependencies=[]）
- dependencies 用 id 字符串引用，如 ["task-1"]
- 不要在 JSON 外面加 ``` 标记或解释文字
- 严格 JSON，字符串用双引号"""


def extract_json_block(content: str) -> str | None:
    """从 LLM 响应中抽取 JSON 子串。"""
    if not content:
        return None
    content = content.strip()

    m = re.search(r"```json\s*([\s\S]+?)```", content)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*([\s\S]+?)```", content)
    if m:
        return m.group(1).strip()

    # 平衡匹配
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(content):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return content[start : i + 1]
    return None


def parse_planner_response(content: str) -> PlannerOutput | None:
    """单次 JSON + Pydantic 解析，返回 PlannerOutput 或 None。"""
    json_str = extract_json_block(content)
    if not json_str:
        return None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.debug("JSON decode failed", error=str(e)[:100])
        return None
    try:
        return PlannerOutput.model_validate(data)
    except ValidationError as e:
        logger.debug("Pydantic validation failed", error=str(e)[:300])
        return None


async def parse_plan_with_reask(
    llm: Any,
    user_prompt: str,
    *,
    context_str: str = "",
    max_attempts: int | None = None,
    timeout: float = 60.0,
    available_agents_desc: str = "",
) -> PlannerOutput | None:
    """Plan 解析 + 拓扑校验 + LLM 自纠。

    流程：
    1. LLM 调用 → JSON 解析 → Pydantic 校验
    2. 拓扑校验（validate_plan）
    3. 有问题 → 把 issues 注入 reask prompt → 再次 LLM 调用
    4. 重复 max_attempts 次
    5. 全失败 → 返回 None（caller 走 fallback）
    """
    if max_attempts is None:
        max_attempts = DEFAULT_RELIABILITY_CONFIG["parse_max_attempts"]

    from .registry import all as _all_cards

    schema_json = json.dumps(
        PlannerOutput.model_json_schema(), ensure_ascii=False, indent=2
    )

    if not available_agents_desc:
        available_agents_desc = "\n".join(c.name for c in _all_cards())

    # 第一次调用：只有 system + user prompt
    messages = [
        SystemMessage(
            content=(
                "你是任务规划 Agent。基于用户请求输出严格的 JSON 计划，"
                "schema 如下:\n```json\n" + schema_json + "\n```"
            )
        ),
        HumanMessage(content=user_prompt),
    ]

    last_issues: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt == 1:
                resp = await asyncio.wait_for(
                    llm.ainvoke(messages), timeout=timeout
                )
            else:
                # Reask：把上次的问题注入，让 LLM 在下一轮纠正
                issues_text = "\n".join(f"- {i}" for i in last_issues)
                reask_prompt = REASK_SYSTEM_PROMPT.format(
                    issues=issues_text,
                    schema=schema_json,
                    available_agents=available_agents_desc,
                )
                resp = await asyncio.wait_for(
                    llm.ainvoke(
                        messages + [
                            HumanMessage(content=reask_prompt),
                            HumanMessage(content=f"## 原始用户请求\n{user_prompt}\n\n"
                                f"## 上下文\n{context_str}" if context_str else ""),
                        ]
                    ),
                    timeout=timeout,
                )

            content = resp.content if hasattr(resp, "content") else str(resp)

            # Layer 1: JSON + Pydantic 解析
            parsed = parse_planner_response(content)
            if parsed is None:
                last_issues = ["响应无法解析为合法 JSON，请直接输出 { ... } 格式的 JSON"]
                logger.warning("plan parse failed, will reask", attempt=attempt)
                continue

            # Layer 2: 拓扑校验
            vresult = validate_plan(parsed)
            if vresult.valid:
                logger.info(
                    "plan validated OK on attempt %d, tasks=%d",
                    attempt, len(parsed.tasks),
                )
                return parsed

            # 有校验问题 → 注入 reask prompt，让 LLM 下一轮纠正
            last_issues = vresult.issues
            logger.warning(
                "plan validation issues on attempt %d: %s",
                attempt, vresult.issues,
            )

        except asyncio.TimeoutError:
            last_issues = [f"LLM 调用超时 ({timeout}s)"]
            logger.warning("plan LLM timeout", attempt=attempt)
        except Exception as e:
            err_str = str(e).lower()
            is_permanent = any(
                code in err_str
                for code in (
                    "401", "403", "404",
                    "unauthorized", "forbidden", "not found",
                    "invalid_api_key", "permission denied",
                )
            )
            last_issues = [f"LLM 调用异常: {str(e)[:200]}"]
            logger.warning(
                "plan LLM error",
                attempt=attempt,
                permanent=is_permanent,
                error=last_issues[0],
            )
            if is_permanent:
                break

    logger.error(
        "plan parse exhausted all %d attempts, last_issues=%s",
        max_attempts, last_issues,
    )
    return None
