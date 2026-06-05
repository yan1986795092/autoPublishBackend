"""Centralized logging configuration — 委托给 shared-logging 模块。

This module re-exports shared-logging's config to avoid breaking existing
imports throughout the codebase. All actual configuration is in
src.logging_shared.config.
"""

import structlog

# 委托给 shared-logging
from src.logging_shared.config import setup, get_logger as _sl_get_logger
from src.logging_shared.middleware import LoggingMiddleware


def setup_logging(settings) -> None:
    """Configure structlog + stdlib logging from application Settings.

    Delegates to src.logging_shared.config.setup().
    Reads ``settings.log_level`` and ``settings.log_json_format``.
    """
    setup(
        log_level=getattr(settings, "log_level", "INFO"),
        log_json=getattr(settings, "log_json_format", False),
        service_name="autopublish-api",
    )


def get_logger(name: str | None = None):
    return _sl_get_logger(name)
