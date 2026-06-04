"""GatewayContainer — singleton manager for all shared Gateway components.

Manages lifecycle of:
  - CircuitBreaker (per-model failure isolation)
  - MultiDimensionalRateLimiter (RPM/TPM sliding windows)
  - CostTracker (SQLite-backed usage recording)
  - LatencyTracker (P95 latency feedback for EnhancedRouter)
  - LiteLLMClient (actual LLM calls with OTel spans)

Usage:
    # In app lifespan / worker.start():
    container = await GatewayContainer.get_instance()

    # In ModelRouter (sync init):
    container = GatewayContainer.get_instance_sync()
    candidates = container.build_candidates_from_settings(settings)
"""

from __future__ import annotations

import asyncio
from typing import Optional

import structlog
from .circuit_breaker import CircuitBreaker
from .rate_limiter import MultiDimensionalRateLimiter, SlidingWindowRateLimiter
from .cost_tracker import CostTracker
from .router import ModelCandidate, LatencyTracker
from ..litellm_client import LiteLLMClient, GatewayCallbacks
from ..config import get_gateway_settings, GatewaySettings

logger = structlog.get_logger(__name__)


class GatewayContainer:
    """Singleton manager for Gateway shared components.

    All stateful components (circuit breaker, rate limiter, latency tracker,
    cost tracker) live here as singletons so that all GatewaySession instances
    share the same state — circuit breaker state accumulates across all calls,
    rate limiter windows are global, P95 latency is aggregated, etc.
    """

    _instance: Optional["GatewayContainer"] = None
    _lock = asyncio.Lock()

    def __init__(self, settings: Optional[GatewaySettings] = None):
        self.settings = settings or get_gateway_settings()

        # Latency tracker — shared P95 feedback for EnhancedRouter
        self.latency_tracker = LatencyTracker()

        # Circuit breaker — per-model failure isolation
        self.circuit_breaker = CircuitBreaker(
            provider="all",
            failure_threshold=self.settings.cb_failure_threshold,
            error_rate_threshold=self.settings.cb_error_rate_threshold,
            window_seconds=self.settings.cb_window_seconds,
            cooldown_seconds=self.settings.cb_cooldown_seconds,
            half_open_probe_count=self.settings.cb_half_open_probe_count,
            health_check_interval=self.settings.cb_health_check_interval,
            health_check_timeout=self.settings.cb_health_check_timeout,
            health_check_model=self.settings.cb_health_check_model,
        )

        # Register with OTel ObservableGauge callback
        from src.observability.metrics import set_circuit_breaker
        set_circuit_breaker(self.circuit_breaker)

        # Rate limiter — sliding window RPM/TPM per provider + per source app
        self.rate_limiter = MultiDimensionalRateLimiter(
            provider_limits=self.settings.provider_limits,
            source_app_limits=self.settings.source_app_limits,
            timeout_ms=self.settings.queue_timeout_ms,
            backend=SlidingWindowRateLimiter(),
        )

        # Cost tracker — SQLite-backed async batching
        self.cost_tracker = CostTracker(
            db_path=self.settings.cost_db_path,
            batch_size=self.settings.cost_batch_size,
            flush_interval=self.settings.cost_flush_interval,
            pricing=self.settings.pricing,
        )

        # LiteLLM client — wraps litellm with OTel + callbacks
        self.litellm_client = LiteLLMClient(
            success_callback=GatewayCallbacks(
                latency_tracker=self.latency_tracker,
                circuit_breaker=self.circuit_breaker,
            ),
            failure_callback=GatewayCallbacks(
                circuit_breaker=self.circuit_breaker,
            ),
            circuit_breaker=self.circuit_breaker,
            rate_limiter=self.rate_limiter,
        )

        self._started = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        """Start background tasks: circuit breaker health check, cost tracker flush."""
        if self._started:
            return
        self._started = True
        await self.circuit_breaker.start()
        await self.cost_tracker.start()
        logger.info("GatewayContainer started", component="gateway", lifecycle="start")

    async def stop(self):
        """Gracefully stop all background tasks."""
        if not self._started:
            return
        self._started = False
        await self.circuit_breaker.stop()
        await self.cost_tracker.stop()
        logger.info("GatewayContainer stopped", component="gateway", lifecycle="stop")

    # ── Singleton access ─────────────────────────────────────────────────

    @classmethod
    async def get_instance(cls) -> "GatewayContainer":
        """Get or create the singleton, auto-starting background tasks.

        Use this in async contexts: app lifespan, worker.start(), etc.
        """
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    instance = cls()
                    await instance.start()
                    cls._instance = instance
        return cls._instance

    @classmethod
    def get_instance_sync(cls) -> "GatewayContainer":
        """Synchronous getter for contexts where async init is not possible.

        Does NOT auto-start background tasks. The caller is responsible for
        ensuring start() is called at an appropriate lifecycle point
        (e.g., FastAPI lifespan or AgentWorker.start()).

        Used by ModelRouter which is initialized synchronously in factory.py.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Candidate builder ────────────────────────────────────────────────

    def build_candidates_from_settings(self, main_settings) -> list[ModelCandidate]:
        """Build ModelCandidate list from the main application Settings.

        Reads all configured providers and creates a ModelCandidate for each
        that has an API key configured. This mirrors _build_router() in
        api/chat.py but works with the main Settings object.

        Args:
            main_settings: The main app Settings (src.config.get_settings()).

        Returns:
            List of ModelCandidate entries for all available providers.
        """
        candidates = []
        pricing = self.settings.pricing

        # Default static latency estimates per provider (ms)
        static_latency_map = {
            "openai": 1000,
            "zhipu": 800,
            "qwen": 1200,
        }

        for provider_key, provider_name in [("openai", "openai"), ("zhipu", "zhipuai"), ("qwen", "qwen")]:
            api_key = getattr(main_settings, f"{provider_key}_api_key", None)
            if not api_key:
                continue
            model = getattr(main_settings, f"{provider_key}_model", "") or ""
            base_url = getattr(main_settings, f"{provider_key}_base_url", "") or ""

            candidates.append(ModelCandidate(
                provider=provider_name,
                model=model,
                api_key=api_key,
                base_url=base_url,
                input_price=pricing.get(model, {}).get("input", 0.002),
                output_price=pricing.get(model, {}).get("output", 0.002),
                static_latency_ms=static_latency_map.get(provider_key, 1000),
            ))

        logger.info(
            "GatewayContainer: built %d candidates from settings (%s)",
            len(candidates),
            ", ".join(c.provider for c in candidates),
        )
        return candidates
