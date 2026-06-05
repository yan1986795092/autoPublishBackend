"""统一 structlog 配置（两服务共用）"""

import logging
import sys
from typing import Literal

import structlog

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
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
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
        _l.setLevel(level)

    for _name in ["elasticsearch", "elastic_transport", "opentelemetry",
                  "urllib3", "sqlalchemy.engine", "httpx", "httpcore"]:
        logging.getLogger(_name).setLevel(logging.WARNING)


def _add_service_context(logger, method_name, event_dict):
    """注入 service_name 到每个日志事件。"""
    event_dict["service"] = event_dict.get("service", "unknown")
    return event_dict


def get_logger(name: str | None = None):
    """获取 logger 实例。"""
    if name:
        return structlog.get_logger().bind(logger_name=name)
    return structlog.get_logger()
