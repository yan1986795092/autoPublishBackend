import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
from .circuit_breaker import CircuitBreaker


@dataclass
class ModelCandidate:
    provider: str
    model: str
    api_key: str
    base_url: str
    input_price: float
    output_price: float
    static_latency_ms: float


class LatencyTracker:
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._latencies: dict[str, deque] = {}
        self._lock = asyncio.Lock()

    async def record(self, model: str, latency_ms: float):
        async with self._lock:
            if model not in self._latencies:
                self._latencies[model] = deque()
            latencies = self._latencies[model]
            now = time.time()
            latencies.append((now, latency_ms))
            cutoff = now - self.window_seconds
            while latencies and latencies[0][0] < cutoff:
                latencies.popleft()

    async def get_p95(self, model: str) -> Optional[float]:
        async with self._lock:
            latencies = self._latencies.get(model)
            if not latencies:
                return None
            values = sorted([l for _, l in latencies])
            if not values:
                return None
            idx = math.ceil(len(values) * 0.95) - 1
            return values[max(0, min(idx, len(values) - 1))]


class EnhancedRouter:
    def __init__(
        self,
        candidates: list[ModelCandidate],
        circuit_breaker: CircuitBreaker,
        weights: Optional[dict[str, float]] = None,
        hints: Optional[dict[str, dict[str, float]]] = None,
        latency_tracker: Optional[LatencyTracker] = None,
    ):
        self.candidates = candidates
        self.cb = circuit_breaker
        self.weights = weights or {"cost": 0.4, "latency": 0.3, "actual_latency": 0.3}
        self.hints = hints or {}
        self.latency_tracker = latency_tracker or LatencyTracker()

    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        if max_val == min_val:
            return 1.0
        return 1.0 - (value - min_val) / (max_val - min_val)

    async def select(self, hint: Optional[str] = None) -> Optional[ModelCandidate]:
        weights = self.hints.get(hint, self.weights)

        available = [c for c in self.candidates if self.cb.is_available(c.model)]
        if not available:
            return None

        cost_vals = [c.input_price + c.output_price for c in available]
        lat_vals = [c.static_latency_ms for c in available]

        p95_latencies = {}
        for c in available:
            p95 = await self.latency_tracker.get_p95(c.model)
            p95_latencies[c.model] = p95 if p95 else c.static_latency_ms

        max_cost, min_cost = max(cost_vals), min(cost_vals)
        max_lat, min_lat = max(lat_vals), min(lat_vals)
        max_p95 = max(p95_latencies.values())
        min_p95 = min(p95_latencies.values())

        best: Optional[ModelCandidate] = None
        best_score = -1.0

        for c in available:
            norm_cost = self._normalize(c.input_price + c.output_price, min_cost, max_cost)
            norm_lat = self._normalize(c.static_latency_ms, min_lat, max_lat)
            norm_p95 = self._normalize(p95_latencies[c.model], min_p95, max_p95)

            score = (
                weights["cost"] * norm_cost
                + weights["latency"] * norm_lat
                + weights["actual_latency"] * norm_p95
            )
            if score > best_score:
                best_score = score
                best = c

        return best

    async def record_success(self, model: str, latency_ms: float):
        await self.latency_tracker.record(model, latency_ms)
        await self.cb.record_success(model)

    async def record_failure(self, model: str):
        await self.cb.record_failure(model)
