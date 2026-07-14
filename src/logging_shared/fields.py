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

# 渲染类型（RichConsoleRenderer 分派依据）
TYPE = "_type"
TYPE_HTTP = "http"
TYPE_AGENT = "agent"
TYPE_LLM = "llm"
TYPE_DB = "db"
TYPE_SYSTEM = "system"

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

# P1.1 新增：OTel 对齐字段
CLIENT_IP = "client_ip"
TENANT_ID = "tenant_id"
SESSION_ID = "session_id"
BUSINESS_CODE = "business_code"
USER_AGENT = "user_agent"
REFERER = "referer"
TASK_ID = "task_id"
ERROR_CODE = "error_code"
REQUEST_SIZE = "request_size"
RESPONSE_SIZE = "response_size"
DECISION_REASON = "decision_reason"

# 事件类型
LLM_REQUEST_START = "llm_request_start"
LLM_REQUEST_END = "llm_request_end"
HTTP_REQUEST_START = "http_request_start"
HTTP_REQUEST_END = "http_request_end"
HTTP_REQUEST_ERROR = "http_request_error"
