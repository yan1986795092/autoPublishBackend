import logging
import os
import sys
from typing import Literal

import structlog

from .rich_console import RichConsoleRenderer
from .dispatcher import Dispatcher, set_dispatcher
from .sinks.console import ConsoleSink as NewConsoleSink
from .sinks.es import EsSink
from .sinks.jsonfile import JsonFileSink
from .redactor import DefaultRedactor


class _DefaultColorFormatter(logging.Formatter):
    """为纯文本日志添加 ANSI 白色前景，避免终端默认色（可能偏红）。

    structlog 渲染的日志已自带 ANSI 颜色（以 ``\\033[`` 开头），
    这类消息原样通行。纯文本消息（如 uvicorn 日志）用 ANSI 白色 ``\\033[37m`` 包裹。
    在每条日志末尾输出 ``\\033[0m`` 重置，但立即追加 ``\\033[37m`` 预备下一行。
    """

    ANSI_WHITE = "\033[37m"
    ANSI_RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if msg.startswith("\033["):
            return msg  # structlog 渲染，已有颜色
        return f"{self.ANSI_WHITE}{msg}{self.ANSI_RESET}{self.ANSI_WHITE}"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

_configured = False


def setup(
    log_level: LogLevel = "INFO",
    log_json: bool = False,
    service_name: str = "unknown",
) -> None:
    """配置 structlog（两服务启动时各调用一次）。"""
    global _configured
    if _configured:
        return
    _configured = True

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)

    # 清除 root logger 默认的 StreamHandler（会输出原始格式日志）
    for h in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(h)

    # 配置 root logger handler — 只输出 message，structlog 事件已自带格式化。
    # 使用 DefaultColorFormatter 让未着色的纯文本日志（如 uvicorn）也显示为白色。
    _root_handler = logging.StreamHandler(sys.stderr)
    _root_handler.setFormatter(_DefaultColorFormatter("%(message)s"))
    logging.getLogger().addHandler(_root_handler)

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        _add_service_context,
    ]

    if log_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            RichConsoleRenderer(colors=True)
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    for _name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        _l = logging.getLogger(_name)
        if not _l.handlers:
            _h = logging.StreamHandler(sys.stderr)
            _h.setFormatter(logging.Formatter("%(message)s"))
            _l.addHandler(_h)
        _l.setLevel(logging.WARNING)
        _l.propagate = False

    for _name in ["elasticsearch", "elastic_transport", "opentelemetry",
                  "urllib3", "sqlalchemy.engine", "httpx", "httpcore"]:
        logging.getLogger(_name).setLevel(logging.WARNING)

    # ── Phase 3: 同时装配新 Dispatcher (默认启用, 用 LOG_NEW_DISPATCHER_DISABLED 选择退出) ──
    if os.getenv("LOG_NEW_DISPATCHER_DISABLED", "false").lower() not in ("true", "1", "yes"):
        new_sinks = []
        from .sinks.console import ConsoleSink as _CSink
        _console = _CSink(colors=not log_json)
        new_sinks.append(_console)
        if os.getenv("LOG_ES_ENABLED", "true").lower() not in ("false", "0", "no"):
            new_sinks.append(EsSink(index_type="logs"))
        if os.getenv("LOG_JSONFILE_ENABLED", "false").lower() in ("true", "1", "yes"):
            new_sinks.append(JsonFileSink())
        set_dispatcher(Dispatcher(sinks=new_sinks, redactor=DefaultRedactor()))
        _log = logging.getLogger("logging.config")
        _log.info("新 Dispatcher 已启用 (sinks=%d)", len(new_sinks))


def _add_service_context(logger, method_name, event_dict):
    """注入 service_name 到每个日志事件。"""
    event_dict["service"] = event_dict.get("service", "unknown")
    return event_dict


def get_logger(name: str | None = None):
    """获取 logger 实例（延迟绑定，确保 setup() 之后才创建 BoundLogger）。"""
    if name:
        # 使用 initial_values 而不是 .bind()，避免在 setup() 前创建 BoundLogger
        return structlog.get_logger(logger_name=name)
    return structlog.get_logger()
