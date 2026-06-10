"""Observability: tracer/meter for in-process span context only.

HTTP spans are handled by LoggingMiddleware which writes directly to
ES traces-* via UnifiedEsWriter — no OTel exporter pipeline involved.

The OTel SDK is initialized here WITHOUT any exporter. It provides
``start_as_current_span()`` for agent code to use as in-process span context.
Without this initialization, ``opentelemetry.trace.get_tracer()`` still works
but uses a global no-op tracer by default — which is fine for production.
We keep the explicit init for clarity and to set service name metadata.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import (
    Resource,
    SERVICE_NAME,
    SERVICE_VERSION,
    DEPLOYMENT_ENVIRONMENT,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

logger = logging.getLogger(__name__)

_tracer: Optional[Tracer] = None
_initialized = False


def setup_observability(
    service_name: str = "autopublish-api",
    service_version: str = "2.0.0",
    settings=None,
) -> tuple[Optional[Tracer], None]:
    """Initialize tracer provider (no exporter — in-process span context only).

    HTTP tracing: LoggingMiddleware -> UnifiedEsWriter -> ES (not OTel).
    Agent span context: ``get_tracer().start_as_current_span()`` for
    in-process parent/child relationships (not exported anywhere).
    """
    global _tracer, _initialized
    if _initialized:
        return _tracer or trace.get_tracer(service_name), None

    if settings is not None:
        from src.core.logging import setup_logging
        setup_logging(settings)
    else:
        from src.config import Settings
        from src.core.logging import setup_logging
        setup_logging(Settings(log_level="DEBUG", log_json_format=False))

    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        DEPLOYMENT_ENVIRONMENT: os.getenv("DEPLOYMENT_ENVIRONMENT", "development"),
    })

    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    _tracer = trace.get_tracer(service_name, service_version)
    _initialized = True

    logger.info(
        "Tracer initialized (no exporter — in-process context only) service=%s",
        service_name,
    )
    return _tracer, None


def setup_local_dev(settings=None) -> tuple[Optional[Tracer], None]:
    """Local dev mode: tracer provider without exporter."""
    global _tracer, _initialized

    if settings is not None:
        from src.core.logging import setup_logging
        setup_logging(settings)
    else:
        from src.config import Settings
        from src.core.logging import setup_logging
        setup_logging(Settings(log_level="DEBUG", log_json_format=False))

    resource = Resource.create({SERVICE_NAME: "autopublish-api-dev"})
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    _tracer = trace.get_tracer("autopublish-api-dev")
    _initialized = True
    return _tracer, None


def get_tracer(name: str = "autopublish") -> Tracer:
    """Get a tracer instance. Uses OTel global if not explicitly initialized."""
    global _tracer
    if _tracer is not None:
        return _tracer
    return trace.get_tracer(name)


def get_meter(name: str = "autopublish"):
    """Stub: meters not configured (no exporter)."""
    return None
