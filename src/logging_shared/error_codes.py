"""标准化错误码 — Agent / 工具 / LLM 各层通用。

所有错误码是 ``str`` 的子类，可直接赋值给 ``AgentEvent.error_type``
或 ``LogRecord.fields["error_code"]``，无需额外转换。

用法::

    from src.logging_shared.error_codes import ErrorCode

    # 直接比较
    if error == ErrorCode.TOOL_TIMEOUT:
        ...

    # str 序列化（JSON 友好）
    code_str = str(ErrorCode.LLM_RATE_LIMIT)  # "LLM_RATE_LIMIT"
"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """标准化错误码枚举（str 子类，JSON 序列化无需特殊处理）。

    命名约定: {领域}_{具体错误}，全大写，下划线分隔。
    """

    # ── LLM 错误 ──────────────────────────────────────────
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    """调用频率触发限流。"""
    LLM_TIMEOUT = "LLM_TIMEOUT"
    """LLM 请求超时。"""
    LLM_INVALID_RESPONSE = "LLM_INVALID_RESPONSE"
    """LLM 返回了无法解析的响应。"""
    LLM_CONTENT_FILTER = "LLM_CONTENT_FILTER"
    """LLM 内容被安全过滤器拦截。"""
    LLM_AUTH_ERROR = "LLM_AUTH_ERROR"
    """LLM API 认证失败（key 无效/过期）。"""
    LLM_QUOTA_EXCEEDED = "LLM_QUOTA_EXCEEDED"
    """LLM API 配额耗尽。"""

    # ── 工具错误 ──────────────────────────────────────────
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    """工具调用超时。"""
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
    """工具执行时抛出未预期异常。"""
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    """请求的工具不存在。"""
    TOOL_AUTH_ERROR = "TOOL_AUTH_ERROR"
    """工具所需的外部认证失败。"""
    TOOL_RATE_LIMIT = "TOOL_RATE_LIMIT"
    """工具调用频率受限。"""
    TOOL_INVALID_INPUT = "TOOL_INVALID_INPUT"
    """工具收到的入参校验不通过。"""

    # ── Agent 错误 ───────────────────────────────────────
    AGENT_INPUT_INVALID = "AGENT_INPUT_INVALID"
    """Agent 输入校验失败（必填字段缺失或类型错误）。"""
    AGENT_PLAN_FAILED = "AGENT_PLAN_FAILED"
    """Agent 无法生成有效的执行计划。"""
    AGENT_MAX_RETRIES = "AGENT_MAX_RETRIES"
    """Agent 达到最大重试次数后仍然失败。"""
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    """整个 Agent 执行超时。"""
    AGENT_MEMORY_ERROR = "AGENT_MEMORY_ERROR"
    """记忆读写失败。"""

    # ── 授权 / 权限 ──────────────────────────────────────
    AUTH_INVALID = "AUTH_INVALID"
    """认证无效。"""
    AUTH_EXPIRED = "AUTH_EXPIRED"
    """认证已过期。"""
    AUTH_INSUFFICIENT = "AUTH_INSUFFICIENT"
    """权限不足。"""

    # ── 系统 ─────────────────────────────────────────────
    INTERNAL_ERROR = "INTERNAL_ERROR"
    """未预期的内部错误。"""
    CONFIG_ERROR = "CONFIG_ERROR"
    """配置错误。"""
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    """外部依赖不可用（ES / DB / API）。"""
