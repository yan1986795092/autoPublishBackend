"""Observability: lightweight tracing setup (OTel-free).

Agent tracing is handled by `AgentTracer` (see agent_tracer.py) which writes
directly to UnifiedEsWriter in Jaeger-compatible trace format — no Jaeger server needed.

HTTP tracing is handled by LoggingMiddleware → UnifiedEsWriter → ES.

This module provides only `setup_observability()` for service initialization
and `get_tracer()` for backward compatibility.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from src.logging_shared.unified_writer import UnifiedEsWriter

logger = logging.getLogger(__name__)


def setup_observability(
    service_name: str = "autopublish-api",
    service_version: str = "2.0.0",
    settings=None,
) -> tuple[None, None]:
    """Initialize observability infrastructure.

    - Ensures UnifiedEsWriter singleton is started (traces/events/logs → ES)
    - Configures structlog (if settings provided)

    Returns (None, None) — no OTel tracer; use AgentTracer for agent spans.
    """
    if settings is not None:
        from src.logging_shared.config import setup as setup_logging
        setup_logging(settings)
    else:
        from src.config import Settings
        from src.logging_shared.config import setup as setup_logging
        setup_logging(Settings(log_level="DEBUG", log_json_format=False))

    # UnifiedEsWriter is started in main.py lifespan — just ensure it's initialized
    UnifiedEsWriter.get_instance()

    logger.info(
        "Observability initialized (trace writer: UnifiedEsWriter) service=%s",
        service_name,
    )
    return None, None


def setup_local_dev(settings=None) -> tuple[None, None]:
    """Local dev mode: same as setup_observability, no OTel."""
    if settings is not None:
        from src.logging_shared.config import setup as setup_logging
        setup_logging(settings)
    else:
        from src.config import Settings
        from src.logging_shared.config import setup as setup_logging
        setup_logging(Settings(log_level="DEBUG", log_json_format=False))

    UnifiedEsWriter.get_instance()
    logger.info("Local dev observability initialized")
    return None, None


def get_tracer(name: str = "autopublish"):
    """Backward-compatible stub. Use AgentTracer for actual tracing.

    Args:
        name: Ignored (kept for API compatibility).
    Returns: None (agent tracing uses AgentTracer directly).
    """
    from src.observability.agent_tracer import get_agent_tracer
    return get_agent_tracer()


def get_meter(name: str = "autopublish"):
    """Stub: meters not configured."""
    return None
