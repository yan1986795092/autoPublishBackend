"""Observability module: unified tracing, metrics, and structured logging.

Usage in your FastAPI lifespan:

    from src.observability import setup_observability

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tracer, meter = setup_observability(
            service_name="autopublish-api",
            log_json=os.getenv("LOG_JSON", "true").lower() == "true",
        )
        from src.observability import metrics as obs_metrics
        obs_metrics.create_meters()
        yield

Or for local development without Docker:

    from src.observability import setup_local_dev
    tracer, meter = setup_local_dev()
"""

from .tracing import (
    setup_observability,
    setup_local_dev,
    get_tracer,
    get_meter,
)
from . import metrics
from . import es_client

__all__ = [
    "setup_observability",
    "setup_local_dev",
    "get_tracer",
    "get_meter",
    "metrics",
    "es_client",
]
