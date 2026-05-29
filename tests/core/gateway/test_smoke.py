import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.core.gateway.core.cost_tracker import CostTracker, CostRecord
from src.core.gateway.core.circuit_breaker import CircuitBreaker, CircuitState
from src.core.gateway.core.rate_limiter import SlidingWindowRateLimiter, MultiDimensionalRateLimiter
from src.core.gateway.core.router import EnhancedRouter, ModelCandidate, LatencyTracker
from src.core.gateway.core.security import validate_request_body, mask_message
from datetime import datetime


@pytest.mark.asyncio
async def test_cost_tracker_calculate_cost():
    tracker = CostTracker(db_path=":memory:", pricing={
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    })
    cost = tracker.calculate_cost("gpt-3.5-turbo", 1000, 500)
    # 1000/1000 * 0.0015 + 500/1000 * 0.002 = 0.0015 + 0.001 = 0.0025
    assert abs(cost - 0.0025) < 0.0001


@pytest.mark.asyncio
async def test_cost_tracker_calculate_cost_unknown_model():
    tracker = CostTracker(db_path=":memory:", pricing={})
    cost = tracker.calculate_cost("unknown-model", 1000, 500)
    assert cost == 0.0


@pytest.mark.asyncio
async def test_circuit_breaker_closed_on_success():
    cb = CircuitBreaker(provider="test", failure_threshold=3)
    await cb.record_success("model-a")
    assert cb.is_available("model-a") is True
    assert cb.get_state("model-a") == "closed"


@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_consecutive_failures():
    cb = CircuitBreaker(provider="test", failure_threshold=3)
    await cb.record_failure("model-a")
    await cb.record_failure("model-a")
    assert cb.is_available("model-a") is True
    await cb.record_failure("model-a")
    assert cb.is_available("model-a") is False
    assert cb.get_state("model-a") == "open"


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_resets_counters():
    cb = CircuitBreaker(provider="test", failure_threshold=3, cooldown_seconds=0)
    await cb.record_failure("model-a")
    await cb.record_failure("model-a")
    await cb.record_failure("model-a")
    assert cb.get_state("model-a") == "open"
    # Advance time past cooldown
    cb._snapshots["model-a"].opened_at = 0
    cb._snapshots["model-a"].state = CircuitState.OPEN
    # Simulate cooldown passed — is_available should transition to HALF_OPEN
    import time
    cb._snapshots["model-a"].opened_at = time.time() - cb.cooldown_seconds - 1
    available = cb.is_available("model-a")
    assert available is True
    assert cb.get_state("model-a") == "half_open"


@pytest.mark.asyncio
async def test_rate_limiter_allows_under_limit():
    limiter = SlidingWindowRateLimiter()
    result = await limiter.acquire("test-key", rpm=10, tpm=1000, token_count=100, timeout_ms=1000)
    assert result.allowed is True


@pytest.mark.asyncio
async def test_rate_limiter_blocks_over_rpm_limit():
    limiter = SlidingWindowRateLimiter()
    for _ in range(3):
        await limiter.acquire("test-key", rpm=3, tpm=999999, token_count=100, timeout_ms=100)
    result = await limiter.acquire("test-key", rpm=3, tpm=999999, token_count=100, timeout_ms=100)
    assert result.allowed is False


@pytest.mark.asyncio
async def test_rate_limiter_blocks_over_tpm_limit():
    limiter = SlidingWindowRateLimiter()
    await limiter.acquire("test-key", rpm=999, tpm=100, token_count=100, timeout_ms=100)
    result = await limiter.acquire("test-key", rpm=999, tpm=100, token_count=1, timeout_ms=100)
    assert result.allowed is False


@pytest.mark.asyncio
async def test_multi_dimensional_rate_limiter():
    limiter = MultiDimensionalRateLimiter(
        provider_limits={"openai": {"rpm": 10, "tpm": 1000}},
        source_app_limits={"default": {"rpm": 100}},
    )
    result = await limiter.check(provider="openai", source_app="default", token_count=500)
    assert result.allowed is True


@pytest.mark.asyncio
async def test_router_selects_available_model():
    cb = CircuitBreaker(provider="test", failure_threshold=3)
    candidates = [
        ModelCandidate(provider="openai", model="gpt-3.5-turbo", api_key="test",
                      base_url="https://api.openai.com/v1",
                      input_price=0.0015, output_price=0.002, static_latency_ms=1000),
        ModelCandidate(provider="zhipu", model="glm-4-flash", api_key="test",
                      base_url="https://zhipu.ai",
                      input_price=0.0005, output_price=0.0005, static_latency_ms=800),
    ]
    router = EnhancedRouter(candidates=candidates, circuit_breaker=cb)
    selected = await router.select()
    assert selected is not None
    assert selected.model in ["gpt-3.5-turbo", "glm-4-flash"]


@pytest.mark.asyncio
async def test_router_excludes_circuit_open_models():
    cb = CircuitBreaker(provider="test", failure_threshold=1)
    await cb.record_failure("gpt-3.5-turbo")
    candidates = [
        ModelCandidate(provider="openai", model="gpt-3.5-turbo", api_key="test",
                      base_url="https://api.openai.com/v1",
                      input_price=0.0015, output_price=0.002, static_latency_ms=1000),
        ModelCandidate(provider="zhipu", model="glm-4-flash", api_key="test",
                      base_url="https://zhipu.ai",
                      input_price=0.0005, output_price=0.0005, static_latency_ms=800),
    ]
    router = EnhancedRouter(candidates=candidates, circuit_breaker=cb)
    selected = await router.select()
    assert selected is not None
    assert selected.model == "glm-4-flash"


@pytest.mark.asyncio
async def test_router_hint_simple_prefers_cheap():
    cb = CircuitBreaker(provider="test", failure_threshold=3)
    candidates = [
        ModelCandidate(provider="openai", model="gpt-4", api_key="test",
                      base_url="https://api.openai.com/v1",
                      input_price=0.03, output_price=0.06, static_latency_ms=2000),
        ModelCandidate(provider="zhipu", model="glm-4-flash", api_key="test",
                      base_url="https://zhipu.ai",
                      input_price=0.0005, output_price=0.0005, static_latency_ms=800),
    ]
    router = EnhancedRouter(
        candidates=candidates,
        circuit_breaker=cb,
        hints={"simple": {"cost": 0.9, "latency": 0.05, "actual_latency": 0.05}},
    )
    selected = await router.select(hint="simple")
    assert selected is not None
    assert selected.model == "glm-4-flash"


@pytest.mark.asyncio
async def test_router_returns_none_when_all_unavailable():
    cb = CircuitBreaker(provider="test", failure_threshold=1)
    await cb.record_failure("gpt-3.5-turbo")
    await cb.record_failure("glm-4-flash")
    candidates = [
        ModelCandidate(provider="openai", model="gpt-3.5-turbo", api_key="test",
                      base_url="https://api.openai.com/v1",
                      input_price=0.0015, output_price=0.002, static_latency_ms=1000),
        ModelCandidate(provider="zhipu", model="glm-4-flash", api_key="test",
                      base_url="https://zhipu.ai",
                      input_price=0.0005, output_price=0.0005, static_latency_ms=800),
    ]
    router = EnhancedRouter(candidates=candidates, circuit_breaker=cb)
    selected = await router.select()
    assert selected is None


@pytest.mark.asyncio
async def test_latency_tracker_p95_calculation():
    tracker = LatencyTracker(window_seconds=60)
    # 20 samples: index 0-19, P95 = index 18 (ceil(20*0.95)-1 = 19-1 = 18)
    latencies = [100.0 + i * 10 for i in range(20)]
    for lat in latencies:
        await tracker.record("model-a", lat)
    p95 = await tracker.get_p95("model-a")
    # Sorted values are 100, 110, ..., 290. P95 = value at index 18 = 280
    assert p95 == 280.0


@pytest.mark.asyncio
async def test_latency_tracker_empty_returns_none():
    tracker = LatencyTracker()
    p95 = await tracker.get_p95("unknown-model")
    assert p95 is None


def test_validate_request_body_missing_model():
    error = validate_request_body({}, set(), 32768)
    assert error == "model field is required"


def test_validate_request_body_unknown_model():
    error = validate_request_body({"model": "unknown"}, {"gpt-3.5-turbo"}, 32768)
    assert "not in the allowed list" in error


def test_validate_request_body_content_too_long():
    error = validate_request_body(
        {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "x" * 40000}]},
        {"gpt-3.5-turbo"},
        32768,
    )
    assert "exceeds" in error


def test_mask_message():
    result = mask_message("Hello, world!")
    assert result.startswith("Hello, world!...")
    assert "[hash:" in result
    assert len(result) < 50
