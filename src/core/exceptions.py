"""Custom exception hierarchy for AutoPublish API.

Architecture:
  - APIException: base for all client-facing API errors (4xx)
  - ServiceError: base for service-layer errors (Agent, Generate, Config)
  - SystemError: base for system errors (not exposed to clients)

All exception classes expose: message, details, and their own error_code.
The exception handler layer maps these to HTTP status codes.
"""

from __future__ import annotations


class APIException(Exception):
    """Base class for all client-facing API errors."""

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    default_message: str = "An internal error occurred"

    def __init__(self, message: str | None = None, details: dict | None = None):
        self.message = message or self.default_message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(APIException):
    """422 — One or more request parameters failed validation."""
    status_code = 422
    error_code = "VALIDATION_ERROR"
    default_message = "Request validation failed"


class AuthenticationError(APIException):
    """401 — Authentication is required or has failed."""
    status_code = 401
    error_code = "AUTHENTICATION_REQUIRED"
    default_message = "Authentication required"


class AuthorizationError(APIException):
    """403 — Authenticated but lacks required permissions."""
    status_code = 403
    error_code = "PERMISSION_DENIED"
    default_message = "Permission denied"


class NotFoundError(APIException):
    """404 — The requested resource does not exist."""
    status_code = 404
    error_code = "NOT_FOUND"
    default_message = "Resource not found"


class ConflictError(APIException):
    """409 — The request conflicts with the current state of the resource."""
    status_code = 409
    error_code = "CONFLICT"
    default_message = "Resource conflict"


class RateLimitError(APIException):
    """429 — The client has sent too many requests."""
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"
    default_message = "Rate limit exceeded"


# ─────────────────────────────────────────────────────────────────────────────
# Service layer exceptions
# ─────────────────────────────────────────────────────────────────────────────


class ServiceError(Exception):
    """Base class for service-layer errors (Agent, Generate, Config)."""

    status_code: int = 500
    error_code: str = "SERVICE_ERROR"
    default_message: str = "Service error"

    def __init__(self, message: str | None = None, details: dict | None = None):
        self.message = message or self.default_message
        self.details = details or {}
        super().__init__(self.message)


class GenerateError(ServiceError):
    """Content generation failed (e.g. LLM returned invalid output)."""
    status_code = 500
    error_code = "GENERATION_ERROR"
    default_message = "Content generation failed"


class AgentError(ServiceError):
    """Base class for agent execution errors."""
    error_code = "AGENT_ERROR"
    default_message = "Agent execution failed"


class HandlerError(AgentError):
    """A handler (prompt node) failed during execution."""
    error_code = "HANDLER_ERROR"
    default_message = "Handler execution failed"

    def __init__(self, message: str | None = None, handler: str | None = None, details: dict | None = None):
        self.handler = handler
        if handler:
            details = {**(details or {}), "handler": handler}
        super().__init__(message, details=details)


class ToolExecutionError(AgentError):
    """An MCP tool call failed during execution."""
    error_code = "TOOL_EXECUTION_ERROR"
    default_message = "Tool execution failed"


class ConfigurationError(AgentError):
    """A configuration value is missing or invalid."""
    error_code = "CONFIGURATION_ERROR"
    default_message = "Configuration error"


# ─────────────────────────────────────────────────────────────────────────────
# System layer exceptions
# ─────────────────────────────────────────────────────────────────────────────


class SystemError(Exception):
    """Base class for system errors (not exposed to clients)."""
    pass


class DatabaseError(SystemError):
    """A database operation failed."""
    pass


class ExternalServiceError(SystemError):
    """An external service (e.g. Elasticsearch, queue) is unavailable."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Shared response builder (used by exception handlers in main.py)
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
    """Build the shared error response payload dict.

    Args:
        exc: The exception instance.
        request_or_headers: Either a FastAPI Request (has .headers) or a
            plain dict of headers. Used to extract trace_id and CORS origin.
        status_code: HTTP status code.
        error_code: Stable error code string for client branching.
        details: Optional extra data (field errors, resource ids, etc.).

    Returns:
        A dict ready to be JSON-serialised as the response body.
    """
    # Extract trace_id from request state (set by LoggingMiddleware)
    trace_id = None
    if hasattr(request_or_headers, "state"):
        trace_id = getattr(request_or_headers.state, "trace_id", None)

    # Extract CORS origin from request headers
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

    # Attach Retry-After header for rate limit errors
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
