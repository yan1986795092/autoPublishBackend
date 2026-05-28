"""Centralized logging configuration for the agent system."""

import logging
import sys
from typing import Optional


_configured = False


def configure_logging(debug: bool = False, json_format: bool = False) -> None:
    """Configure root logger with handler, formatter, and level.

    Safe to call multiple times — only the first call takes effect.
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    level = logging.DEBUG if debug else logging.INFO
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    if json_format:
        try:
            import structlog
            structlog.configure(
                processors=[
                    structlog.stdlib.add_log_level,
                    structlog.dev.ConsoleRenderer(),
                ],
                wrapper_class=structlog.stdlib.BoundLogger,
                context_class=dict,
                logger_factory=structlog.PrintLoggerFactory(),
            )
            # structlog handles formatting; the stdlib handler is a fallback.
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )
        except ImportError:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    handler.setFormatter(formatter)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Thin wrapper around logging.getLogger for import consistency.
    """
    return logging.getLogger(name)
