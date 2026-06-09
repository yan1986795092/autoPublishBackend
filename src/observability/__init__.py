"""Observability module: lightweight tracing, metrics, and ES query helpers.

HTTP spans are written by ``LoggingMiddleware`` directly to ES traces-*.
OTel SDK is only used by agent code for in-process span context.

Usage in FastAPI lifespan::

    from src.observability import setup_observability, metrics as obs_metrics

    tracer, meter = setup_observability(
        service_name="autopublish-api",
        service_version="2.0.0",
        settings=settings,
    )
    obs_metrics.create_meters()

For local dev without Docker::

    from src.observability import setup_local_dev
    tracer, meter = setup_local_dev(settings)
"""

from .tracing import (
    setup_observability,
    setup_local_dev,
    get_tracer,
    get_meter,
)
from . import metrics

__all__ = [
    "setup_observability",
    "setup_local_dev",
    "get_tracer",
    "get_meter",
    "metrics",
]
