"""Observability: lightweight setup (no OTel SDK).

HTTP spans are handled by LoggingMiddleware which writes directly to
ES traces-* via UnifiedEsWriter — no OTel pipeline involved.

This module exists only for backward compatibility with main.py imports.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_initialized = False


def setup_observability(
    service_name: str = "autopublish-api",
    service_version: str = "2.0.0",
    settings=None,
) -> tuple[None, None]:
    """Initialize structlog (OTel SDK removed — unused).

    HTTP tracing is handled by LoggingMiddleware -> UnifiedEsWriter -> ES.
    """
    global _initialized
    if _initialized:
        return None, None

    if settings is not None:
        from src.core.logging import setup_logging
        setup_logging(settings)
    else:
        from src.config import Settings
        from src.core.logging import setup_logging
        setup_logging(Settings(log_level="DEBUG", log_json_format=False))

    _initialized = True
    logger.info(
        "Observability ready (direct-to-ES tracing via UnifiedEsWriter) service=%s",
        service_name,
    )
    return None, None


def setup_local_dev(settings=None) -> tuple[None, None]:
    """Local dev mode: structlog only, no OTel SDK."""
    if settings is not None:
        from src.core.logging import setup_logging
        setup_logging(settings)
    else:
        from src.config import Settings
        from src.core.logging import setup_logging
        setup_logging(Settings(log_level="DEBUG", log_json_format=False))

    global _initialized
    _initialized = True
    return None, None


def get_tracer(name: str = "autopublish"):
    """Stub: returns None (OTel SDK removed)."""
    return None


def get_meter(name: str = "autopublish"):
    """Stub: returns None (OTel SDK removed)."""
    return None
