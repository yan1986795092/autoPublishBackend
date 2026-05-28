class GenerateError(Exception):
    """带错误详情的自定义异常，确保 CORS 头正确返回"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class AgentError(Exception):
    """Agent 系统异常基类"""
    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class HandlerError(AgentError):
    """Handler 执行失败"""
    def __init__(self, message: str, handler: str = "", details: dict | None = None):
        self.handler = handler
        super().__init__(message, details=details)


class ToolExecutionError(AgentError):
    """工具执行失败"""
    def __init__(self, message: str, tool_name: str = "", details: dict | None = None):
        self.tool_name = tool_name
        super().__init__(message, details={"tool_name": tool_name, **(details or {})})


class ConfigurationError(AgentError):
    """配置错误"""
    pass
