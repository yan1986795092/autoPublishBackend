import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_fastapi_instrumentator import Instrumentator as PrometheusInstrumentator

from .config import get_gateway_settings
from .tracing import setup_tracing
from .core.circuit_breaker import CircuitBreaker
from .core.rate_limiter import MultiDimensionalRateLimiter, SlidingWindowRateLimiter
from .core.cost_tracker import CostTracker
from .core.router import LatencyTracker
from .core.security import SecurityMiddleware
from .litellm_client import GatewayCallbacks, LiteLLMClient
from .api import chat, models, stats


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_gateway_settings()
    setup_tracing("model-gateway", log_json=settings.log_json)

    cost_tracker = CostTracker(
        db_path=settings.cost_db_path,
        batch_size=settings.cost_batch_size,
        flush_interval=settings.cost_flush_interval,
        pricing=settings.pricing,
    )
    await cost_tracker.start()

    latency_tracker = LatencyTracker()

    cb = CircuitBreaker(
        provider="all",
        failure_threshold=settings.cb_failure_threshold,
        error_rate_threshold=settings.cb_error_rate_threshold,
        window_seconds=settings.cb_window_seconds,
        cooldown_seconds=settings.cb_cooldown_seconds,
        half_open_probe_count=settings.cb_half_open_probe_count,
        health_check_interval=settings.cb_health_check_interval,
        health_check_timeout=settings.cb_health_check_timeout,
        health_check_model=settings.cb_health_check_model,
    )
    await cb.start()

    rate_limiter = MultiDimensionalRateLimiter(
        provider_limits=settings.provider_limits,
        source_app_limits=settings.source_app_limits,
        timeout_ms=settings.queue_timeout_ms,
        backend=SlidingWindowRateLimiter(),
    )

    success_cb = GatewayCallbacks(
        on_success=None,
        on_failure=None,
        latency_tracker=latency_tracker,
    )
    litellm_client = LiteLLMClient(
        success_callback=success_cb,
        failure_callback=success_cb,
    )

    # Store all state in app.state — accessible from middleware and survives requests.
    app.state.cost_tracker = cost_tracker
    app.state.circuit_breaker = cb
    app.state.rate_limiter = rate_limiter
    app.state.latency_tracker = latency_tracker
    app.state.litellm_client = litellm_client

    yield

    await cb.stop()
    await cost_tracker.stop()


def create_gateway_app() -> FastAPI:
    settings = get_gateway_settings()

    app = FastAPI(
        title="Model Gateway",
        description="OpenAI-compatible LLM gateway with routing, circuit breaking, rate limiting, and cost tracking",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(SecurityMiddleware(
        allowed_source_apps=settings.allowed_source_apps,
        allowed_models=settings.allowed_models,
        max_content_length=settings.max_content_length,
    ))

    @app.middleware("http")
    async def inject_gateway_state(request: Request, call_next):
        # Access app.state — lifespan guarantees this is populated before HTTP handlers run.
        state = getattr(request.app.state, "cost_tracker", None)
        if state is None:
            return await call_next(request)
        request.state.cost_tracker = request.app.state.cost_tracker
        request.state.circuit_breaker = request.app.state.circuit_breaker
        request.state.rate_limiter = request.app.state.rate_limiter
        request.state.latency_tracker = request.app.state.latency_tracker
        request.state.litellm_client = request.app.state.litellm_client
        return await call_next(request)

    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(stats.router)

    FastAPIInstrumentor.instrument_app(app)
    PrometheusInstrumentator().instrument(app).expose(app)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
