"""Agent 标准响应信封 — 所有 handler 统一使用此格式

保证每个 agent 的输出结构一致，下游 _build_dependency_context 无需
再写 isinstance 判断 output 类型。
"""

from __future__ import annotations
from typing import Any, Literal

# ── 标准信封结构 ──

# 所有 handler 必须返回此格式的 dict
# {
#     "status": "completed" | "failed" | "partial",
#     "agent_type": str,           # 如 "WritingAgent"
#     "output": {
#         "content": str,          # 主要文本输出（下游从这读，永远存在）
#         "type": "text",          # "text" | "metadata" | "code" | "diagram" | "error"
#         # ... handler 特有字段（results, sources 等）
#     },
#     "error": str | None,
#     "tokens_used": int,
#     "prompt_tokens": int,
#     "completion_tokens": int,
#     "thinking_steps": list[dict],
# }

StatusType = Literal["completed", "failed", "partial"]
OutputType = Literal["text", "metadata", "code", "diagram", "error"]


def agent_response(
    agent_type: str,
    *,
    status: StatusType = "completed",
    content: str = "",
    output_type: OutputType = "text",
    output_extra: dict | None = None,
    output_fields: dict[str, Any] | None = None,
    error: str | None = None,
    tokens_used: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    thinking_steps: list[dict] | None = None,
    **kwargs: Any,
) -> dict:
    """构建标准 Agent 响应。

    所有 handler 必须通过此函数返回结果，确保格式统一。

    Args:
        agent_type: Agent 类型名
        status: 执行状态
        content: 主要文本输出
        output_type: 输出类型
        output_extra: output 中的额外字段
        output_fields: 命名产出物字段（用于 consume 解析的关键索引）。
                       每个 handler 按 schema 填充此字典，下游 consume
                       解析时从 output_fields 按字段名查找。
                       例: WritingAgent → {"article_body": "...", "article_title": "..."}
        error: 错误信息（status=failed 时必填）
        tokens_used: LLM token 用量
        prompt_tokens: prompt token 数
        completion_tokens: completion token 数
        thinking_steps: 思考步骤列表
        **kwargs: 顶层额外字段（如 prompt_versions、failure_type 等）

    Returns:
        标准信封 dict
    """
    output: dict[str, Any] = {
        "content": content,
        "type": output_type,
    }
    if output_extra:
        output.update(output_extra)
    if output_fields:
        output["output_fields"] = output_fields

    result: dict[str, Any] = {
        "status": status,
        "agent_type": agent_type,
        "output": output,
        "error": error,
        "tokens_used": tokens_used,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "thinking_steps": thinking_steps or [],
    }
    result.update(kwargs)
    return result


def error_response(
    agent_type: str,
    error: str,
    failure_type: str = "unknown",
    status: StatusType = "failed",
    tokens_used: int = 0,
    **kwargs: Any,
) -> dict:
    """构建标准错误响应。

    所有异常路径统一使用此函数，避免每个 except 块自己拼 dict。

    Args:
        agent_type: Agent 类型名
        error: 错误描述
        failure_type: 失败类型（unknown / handler / mcp_tool / timeout）
        status: 状态（默认 failed）
        tokens_used: 已消耗的 token
        **kwargs: 额外字段（error_code, tool_name, handler 等）

    Returns:
        标准错误信封
    """
    return agent_response(
        agent_type,
        status=status,
        output_type="error",
        error=error,
        tokens_used=tokens_used,
        thinking_steps=[{
            "agent": agent_type,
            "action": "失败",
            "reasoning": error[:500],
            "status": "failed",
            "failure_type": failure_type,
        }],
        failure_type=failure_type,
        **kwargs,
    )


def safe_handler_run(agent_type: str, handler_fn, task_input: dict, llm, mcp_tools):
    """安全执行 handler，捕获所有异常返回标准错误响应。

    用法：
        result = await safe_handler_run(agent_type, handler, t_input, llm, _mcp)
    """
    import traceback
    try:
        return handler_fn(task_input, llm, mcp_tools)
    except Exception as e:
        logger = __import__("structlog").get_logger(__name__)
        logger.error(
            "Agent handler 异常",
            agent_type=agent_type,
            error=str(e),
            exc_info=True,
        )
        # 根据异常类型匹配失败类型
        from src.core.exceptions import HandlerError, ToolExecutionError

        if isinstance(e, ToolExecutionError):
            return error_response(
                agent_type,
                error=str(e),
                failure_type="mcp_tool",
                tool_name=e.tool_name,
                error_code=e.error_code,
            )
        elif isinstance(e, HandlerError):
            return error_response(
                agent_type,
                error=str(e),
                failure_type="handler",
                error_code=getattr(e, "error_code", ""),
                handler=e.handler,
            )
        else:
            return error_response(
                agent_type,
                error=str(e),
                failure_type="unknown",
                error_type=type(e).__name__,
            )
