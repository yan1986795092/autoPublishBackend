"""Observability module: lightweight setup.

HTTP trace spans are written by LoggingMiddleware directly to ES traces-*
via UnifiedEsWriter. No OTel SDK is used.
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
