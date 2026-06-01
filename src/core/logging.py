"""Centralized logging configuration for the agent system.

This is THE single entry point for all logging configuration.
Call ``setup_logging(settings)`` once at application startup.
All other modules should use ``import structlog; logger = structlog.get_logger(__name__)``.
"""

import logging
import sys

import structlog

_configured = False


def setup_logging(settings) -> None:
    """Configure structlog + stdlib logging from application Settings.

    Safe to call multiple times — only the first call takes effect.
    Reads ``settings.log_level`` and ``settings.log_json_format``.
    """
    from src.config import Settings  # lazy import to avoid circular deps

    global _configured
    if _configured:
        return
    _configured = True

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)

    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.log_json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route standard library loggers through structlog (only uvicorn for now)
    for _name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        _l = logging.getLogger(_name)
        if not _l.handlers:
            _h = logging.StreamHandler(sys.stderr)
            _h.setFormatter(logging.Formatter("%(message)s"))
            _l.addHandler(_h)
        _l.setLevel(level)
