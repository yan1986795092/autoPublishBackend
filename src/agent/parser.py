"""Structured-output parser with reask loop (Layer 2).

替代 _parse_plan_from_response 的 regex/json.loads 路径, 升级为
Pydantic 校验 + reask。失败时把 Pydantic ValidationError 注入下次的
messages, 让 LLM 自纠。

为什么不引入 Instructor:
- 项目依赖里没有 instructor, 引入需要 review + 测试, 风险大于收益
- 我们用 Pydantic + 自写 reask loop, 30 行就能达到同样效果
- 保留可替换边界: 后续如需切 Instructor, 只改这一个文件

Reference:
- Instructor retry: max_retries + ValidationError 注入 messages
- Pydantic ValidationError: 可序列化, 可注入 prompt
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from pydantic import ValidationError
from langchain_core.messages import SystemMessage, HumanMessage

from .contracts import PlannerOutput, DEFAULT_RELIABILITY_CONFIG
from ..logging_shared import get_logger

logger = get_logger("agent.parser")


REASK_SYSTEM_PROMPT = """你之前的回复无法被解析为合法的 JSON 计划。

请严格按以下 schema 重新输出(只输出 JSON, 不要任何解释/前缀/后缀):

```json
{schema}
```

上一轮的解析错误如下:
{error}

修正要点:
- agent_type 必须是已注册的 Agent 名, 可选: {available_agents}
- 每个 task 必须有 id; dependencies 中的 id 必须能在 plan 的其它 task 里找到
- 必须有至少一个 root task (无 dependency)
- 不要在 JSON 外面加 ``` 标记或解释文字
- 严格 JSON, 字符串用双引号
"""


def extract_json_block(content: str) -> str | None:
    """从 LLM 响应中抽取 JSON 子串。

    优先级:
    1. ```json ... ``` 块
    2. ``` ... ``` 任意代码块
    3. 最外层 {...} 平衡匹配
    """
    if not content:
        return None
    content = content.strip()

    m = re.search(r"```json\s*([\s\S]+?)```", content)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*([\s\S]+?)```", content)
    if m:
        return m.group(1).strip()

    # 平衡匹配 - 跳过字符串内的 { }
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
    """单次 parse: 返回 Pydantic 实例或 None。

    不抛异常, 由 caller 决定是否 reask。
    返回 None 表示解析失败(可能是 JSON 不合法, 或 schema 不匹配)。
    """
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
    """Plan 解析 + 自动 reask。

    流程:
    1. 第一次 LLM 调用
    2. parse_planner_response 校验
    3. 失败 → 拼 reask prompt → 第二次调用(把错误注入)
    4. 重复 max_attempts 次
    5. 全失败 → 返回 None(caller 走 fallback plan)

    Args:
        llm: LangChain ChatModel, 必须支持 ainvoke(messages)
        user_prompt: 第一次的 user prompt(含 user request + intent)
        context_str: 上下文(可选, 注入到第二次起)
        max_attempts: 总尝试次数(含第一次), 默认 DEFAULT_RELIABILITY_CONFIG["parse_max_attempts"]
        timeout: 单次 LLM 调用的超时(秒)
        available_agents_desc: registry 里所有 agent 的描述, 用于 reask 时给 LLM 提示

    Returns:
        校验通过的 PlannerOutput, 或 None(reask 全部失败)
    """
    if max_attempts is None:
        max_attempts = DEFAULT_RELIABILITY_CONFIG["parse_max_attempts"]

    from .registry import format_agent_descriptions, all as _all_cards

    schema_json = json.dumps(
        PlannerOutput.model_json_schema(), ensure_ascii=False, indent=2
    )

    if not available_agents_desc:
        available_agents_desc = ", ".join(c.name for c in _all_cards())

    messages = [
        SystemMessage(
            content=(
                "你是任务规划 Agent。基于用户请求输出严格的 JSON 计划, "
                "schema 如下:\n```json\n" + schema_json + "\n```"
            )
        ),
        HumanMessage(content=user_prompt),
    ]

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt == 1:
                resp = await asyncio.wait_for(
                    llm.ainvoke(messages), timeout=timeout
                )
            else:
                # Reask: 注入错误反馈 + 原始上下文
                reask_msg = HumanMessage(
                    content=REASK_SYSTEM_PROMPT.format(
                        schema=schema_json,
                        error=last_error or "(无法解析为 JSON)",
                        available_agents=available_agents_desc,
                    )
                    + f"\n\n## 原始用户请求\n{user_prompt}\n\n"
                    + (f"## 上下文\n{context_str}\n\n" if context_str else "")
                )
                resp = await asyncio.wait_for(
                    llm.ainvoke(messages + [reask_msg]), timeout=timeout
                )

            content = resp.content if hasattr(resp, "content") else str(resp)
            parsed = parse_planner_response(content)

            if parsed is not None:
                logger.info(
                    "plan parsed OK on attempt",
                    attempt=attempt,
                    tasks=len(parsed.tasks),
                )
                return parsed

            last_error = "JSON 解析失败: 响应无法解析为合法 JSON"
            logger.warning("plan parse failed, will reask", attempt=attempt)

        except asyncio.TimeoutError:
            last_error = f"LLM 调用超时 ({timeout}s)"
            logger.warning("plan LLM timeout", attempt=attempt)
        except Exception as e:
            # FIX #4: 区分 transient (5xx/timeout) vs permanent (401/403/404)
            # permanent error 立刻跳出, 不浪费重试次数
            err_str = str(e).lower()
            is_permanent = any(
                code in err_str
                for code in (
                    "401", "403", "404",  # 鉴权/资源不存在
                    "unauthorized", "forbidden", "not found",
                    "invalid_api_key", "permission denied",
                )
            )
            last_error = f"LLM 调用异常: {str(e)[:200]}"
            logger.warning(
                "plan LLM error",
                attempt=attempt,
                permanent=is_permanent,
                error=last_error,
            )
            if is_permanent:
                logger.error(
                    "plan parse: permanent error 立刻终止 reask",
                    error=last_error,
                )
                break

    logger.error(
        "plan parse exhausted reask",
        max_attempts=max_attempts,
        last_error=last_error,
    )
    return None