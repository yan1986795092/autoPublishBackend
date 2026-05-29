"""Gateway tracing — re-exported from unified observability module.

The gateway now uses src.observability for all tracing/metrics/logging.
This module is kept for backward compatibility.
"""

from src.observability.tracing import get_tracer, setup_observability, setup_local_dev

__all__ = ["get_tracer", "setup_observability", "setup_local_dev"]
