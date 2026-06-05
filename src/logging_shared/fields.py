"""统一日志字段定义（两服务共用）"""

# 顶层标准字段
TRACE_ID = "trace_id"
SPAN_ID = "span_id"
SERVICE_NAME = "service_name"
LOG_TIME = "log_time"
LEVEL = "level"
EVENT = "event"
MESSAGE = "message"
DURATION_MS = "duration_ms"
STATUS_CODE = "status_code"
ERROR_TYPE = "error_type"
ERROR_MESSAGE = "error_message"

# LLM 请求专用字段
LLM_MODEL = "llm_model"
LLM_PROVIDER = "llm_provider"
LLM_PROMPT_TOKENS = "llm_prompt_tokens"
LLM_COMPLETION_TOKENS = "llm_completion_tokens"
LLM_TOTAL_TOKENS = "llm_total_tokens"
LLM_LATENCY_MS = "llm_latency_ms"
LLM_REQUEST_ID = "llm_request_id"
LLM_STREAMING = "llm_streaming"

# 请求上下文字段
REQUEST_METHOD = "request_method"
REQUEST_PATH = "request_path"
REQUEST_BODY_SUMMARY = "request_body_summary"
RESPONSE_BODY_SUMMARY = "response_body_summary"
SOURCE_APP = "source_app"
USER_ID = "user_id"
CONTENT_TYPE = "content_type"

# 事件类型
LLM_REQUEST_START = "llm_request_start"
LLM_REQUEST_END = "llm_request_end"
HTTP_REQUEST_START = "http_request_start"
HTTP_REQUEST_END = "http_request_end"
HTTP_REQUEST_ERROR = "http_request_error"
