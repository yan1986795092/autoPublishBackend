"""AutoPublish API 的自定义异常层次结构。

架构：
  - APIException：所有面向客户端的 API 错误基类（4xx）
  - ServiceError：服务层错误基类（Agent、Generate、Config）
  - SystemError：系统错误基类（不暴露给客户端）

所有异常类暴露：message、details 及其自身的 error_code。
异常处理器层将这些映射为 HTTP 状态码。
"""

from __future__ import annotations


class APIException(Exception):
    """所有面向客户端的 API 错误的基类。"""

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    default_message: str = "发生内部错误"

    def __init__(self, message: str | None = None, details: dict | None = None):
        self.message = message or self.default_message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(APIException):
    """422 — 一个或多个请求参数验证失败。"""
    status_code = 422
    error_code = "VALIDATION_ERROR"
    default_message = "请求验证失败"


class AuthenticationError(APIException):
    """401 — 需要身份验证或身份验证失败。"""
    status_code = 401
    error_code = "AUTHENTICATION_REQUIRED"
    default_message = "需要身份验证"


class AuthorizationError(APIException):
    """403 — 已通过身份验证但缺少所需权限。"""
    status_code = 403
    error_code = "PERMISSION_DENIED"
    default_message = "权限不足"


class NotFoundError(APIException):
    """404 — 请求的资源不存在。"""
    status_code = 404
    error_code = "NOT_FOUND"
    default_message = "资源未找到"


class ConflictError(APIException):
    """409 — 请求与资源的当前状态冲突。"""
    status_code = 409
    error_code = "CONFLICT"
    default_message = "资源冲突"


class RateLimitError(APIException):
    """429 — 客户端发送了太多请求。"""
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"
    default_message = "请求频率超限"


# ─────────────────────────────────────────────────────────────────────────────
# 服务层异常
# ─────────────────────────────────────────────────────────────────────────────


class ServiceError(Exception):
    """服务层错误的基类（Agent、Generate、Config）。"""

    status_code: int = 500
    error_code: str = "SERVICE_ERROR"
    default_message: str = "服务错误"

    def __init__(self, message: str | None = None, details: dict | None = None):
        self.message = message or self.default_message
        self.details = details or {}
        super().__init__(self.message)


class GenerateError(ServiceError):
    """内容生成失败（例如 LLM 返回无效输出）。"""
    status_code = 500
    error_code = "GENERATION_ERROR"
    default_message = "内容生成失败"


class AgentError(ServiceError):
    """Agent 执行错误的基类。"""
    error_code = "AGENT_ERROR"
    default_message = "Agent 执行失败"


class HandlerError(AgentError):
    """处理器（提示节点）在执行期间失败。"""
    error_code = "HANDLER_ERROR"
    default_message = "处理器执行失败"

    def __init__(self, message: str | None = None, handler: str | None = None, details: dict | None = None):
        self.handler = handler
        if handler:
            details = {**(details or {}), "handler": handler}
        super().__init__(message, details=details)


class ToolExecutionError(AgentError):
    """MCP 工具调用在执行期间失败。"""
    error_code = "TOOL_EXECUTION_ERROR"
    default_message = "工具执行失败"


class ConfigurationError(AgentError):
    """配置值缺失或无效。"""
    error_code = "CONFIGURATION_ERROR"
    default_message = "配置错误"


# ─────────────────────────────────────────────────────────────────────────────
# 系统层异常
# ─────────────────────────────────────────────────────────────────────────────


class SystemError(Exception):
    """系统错误的基类（不暴露给客户端）。"""
    pass


class DatabaseError(SystemError):
    """数据库操作失败。"""
    pass


class ExternalServiceError(SystemError):
    """外部服务（例如 Elasticsearch、队列）不可用。"""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 共享响应构建器（由 main.py 中的异常处理器使用）
# ─────────────────────────────────────────────────────────────────────────────

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "http://localhost:5173",
    "Access-Control-Allow-Credentials": "true",
}


def _get_cors_origin(headers: dict | None = None, fallback: str = "http://localhost:5173") -> str:
    if headers:
        origin = headers.get("origin", "")
        allowed = [
            "http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:3000", "http://127.0.0.1:3000",
            "http://localhost:9010", "http://127.0.0.1:9010",
        ]
        if origin in allowed:
            return origin
    return fallback


def build_error_response(
    exc: Exception,
    request_or_headers,
    status_code: int,
    error_code: str,
    details: dict | None = None,
) -> dict:
    """构建共享的错误响应负载字典。

    Args:
        exc: 异常实例。
        request_or_headers: 可以是 FastAPI Request（具有 .headers）或
            简单的 headers 字典。用于提取 trace_id 和 CORS 来源。
        status_code: HTTP 状态码。
        error_code: 用于客户端分支的稳定错误码字符串。
        details: 可选的额外数据（字段错误、资源 ID 等）。

    Returns:
        准备进行 JSON 序列化作为响应体的字典。
    """
    # 从请求状态中提取 trace_id（由 LoggingMiddleware 设置）
    trace_id = None
    if hasattr(request_or_headers, "state"):
        trace_id = getattr(request_or_headers.state, "trace_id", None)

    # 从请求头中提取 CORS 来源
    if hasattr(request_or_headers, "headers"):
        headers_dict = dict(request_or_headers.headers)
    elif isinstance(request_or_headers, dict):
        headers_dict = request_or_headers
    else:
        headers_dict = {}

    cors_origin = _get_cors_origin(headers_dict)
    headers = {
        "Access-Control-Allow-Origin": cors_origin,
        "Access-Control-Allow-Credentials": "true",
    }

    # 为速率限制错误附加 Retry-After 头
    if error_code == "RATE_LIMIT_EXCEEDED" and details:
        retry_after = details.get("retry_after")
        if retry_after:
            headers["Retry-After"] = str(retry_after)

    return {
        "status_code": status_code,
        "content": {
            "success": False,
            "error": {
                "message": getattr(exc, "message", str(exc)),
                "type": type(exc).__name__,
                "code": error_code,
                "details": details or getattr(exc, "details", {}),
                "trace_id": trace_id,
            }
        },
        "headers": headers,
    }
