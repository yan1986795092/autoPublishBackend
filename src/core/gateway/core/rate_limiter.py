import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod


@dataclass
class RateLimitResult:
    allowed: bool
    wait_seconds: float = 0.0
    remaining_rpm: int = 0
    remaining_tpm: int = 0


class RateLimiterBackend(ABC):
    @abstractmethod
    async def acquire(
        self,
        key: str,
        rpm: int,
        tpm: int,
        token_count: int,
        timeout_ms: int,
    ) -> RateLimitResult:
        ...


class SlidingWindowRateLimiter(RateLimiterBackend):
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._request_times: dict[str, deque] = defaultdict(deque)
        self._token_counts: dict[str, deque] = defaultdict(deque)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def acquire(
        self,
        key: str,
        rpm: int,
        tpm: int,
        token_count: int,
        timeout_ms: int,
    ) -> RateLimitResult:
        lock = self._locks[key]
        deadline = time.time() + timeout_ms / 1000

        while True:
            async with lock:
                now = time.time()
                cutoff = now - self.window_seconds

                req_times = self._request_times[key]
                tok_times = self._token_counts[key]

                while req_times and req_times[0] < cutoff:
                    req_times.popleft()
                while tok_times and tok_times[0][0] < cutoff:
                    tok_times.popleft()

                current_rpm = len(req_times)
                current_tpm = sum(c for _, c in tok_times)

                if current_rpm < rpm and current_tpm + token_count <= tpm:
                    req_times.append(now)
                    tok_times.append((now, token_count))
                    return RateLimitResult(
                        allowed=True,
                        remaining_rpm=rpm - current_rpm - 1,
                        remaining_tpm=tpm - current_tpm - token_count,
                    )

            wait_time = 0.1
            remaining = deadline - time.time()
            if remaining <= 0:
                return RateLimitResult(allowed=False, wait_seconds=timeout_ms / 1000)

            await asyncio.sleep(min(wait_time, remaining))


class MultiDimensionalRateLimiter:
    def __init__(
        self,
        provider_limits: dict,
        source_app_limits: dict,
        timeout_ms: int = 5000,
        backend: Optional[RateLimiterBackend] = None,
    ):
        self.provider_limits = provider_limits
        self.source_app_limits = source_app_limits
        self.timeout_ms = timeout_ms
        self.backend = backend or SlidingWindowRateLimiter()

    async def check(
        self,
        provider: str,
        source_app: str,
        token_count: int,
    ) -> RateLimitResult:
        provider_limit = self.provider_limits.get(provider, {"rpm": 999999, "tpm": 999999999})
        rpm = provider_limit.get("rpm", 999999)
        tpm = provider_limit.get("tpm", 999999999)

        result = await self.backend.acquire(
            f"provider:{provider}",
            rpm=rpm,
            tpm=tpm,
            token_count=token_count,
            timeout_ms=self.timeout_ms,
        )
        if not result.allowed:
            return result

        app_limit = self.source_app_limits.get(source_app, self.source_app_limits.get("default", {"rpm": 999999}))
        app_rpm = app_limit.get("rpm", 999999)
        app_result = await self.backend.acquire(
            f"app:{source_app}",
            rpm=app_rpm,
            tpm=999999999,
            token_count=1,
            timeout_ms=self.timeout_ms,
        )
        return app_result
