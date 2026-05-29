import asyncio
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Awaitable, Optional


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitSnapshot:
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    total: int = 0
    last_failure_time: float = 0.0
    opened_at: float = 0.0
    # 半开探测：sent=已放行的探测数，succeeded=成功的探测数
    half_open_probes_sent: int = 0
    half_open_probes_succeeded: int = 0


class CircuitBreaker:
    def __init__(
        self,
        provider: str,
        failure_threshold: int = 3,
        error_rate_threshold: float = 0.3,
        window_seconds: int = 300,
        cooldown_seconds: int = 30,
        half_open_probe_count: int = 3,
        health_check_interval: int = 60,
        health_check_timeout: int = 5,
        health_check_model: str = "gpt-3.5-turbo",
        health_check_fn: Optional[Callable[[], Awaitable[bool]]] = None,
    ):
        self.provider = provider
        self.failure_threshold = failure_threshold
        self.error_rate_threshold = error_rate_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.half_open_probe_count = half_open_probe_count
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.health_check_model = health_check_model
        self.health_check_fn = health_check_fn

        self._snapshots: dict[str, CircuitSnapshot] = {}
        self._events: dict[str, deque] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        self._started = False

    async def start(self):
        if self._started:
            return
        self._started = True
        if self.health_check_fn:
            self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop(self):
        if not self._started:
            return
        self._started = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

    def _get_snapshot(self, model: str) -> CircuitSnapshot:
        if model not in self._snapshots:
            self._snapshots[model] = CircuitSnapshot()
            self._events[model] = deque()
        return self._snapshots[model]

    async def record_success(self, model: str):
        snap = self._get_snapshot(model)
        snap.failures = 0
        now = time.time()
        self._events[model].append((now, True))
        self._prune_events(model, now)

        if snap.state == CircuitState.HALF_OPEN:
            snap.half_open_probes_sent += 1
            snap.half_open_probes_succeeded += 1
            if snap.half_open_probes_succeeded >= self.half_open_probe_count:
                snap.state = CircuitState.CLOSED
                snap.half_open_probes_succeeded = 0
                snap.half_open_probes_sent = 0
                snap.failures = 0

    async def record_failure(self, model: str):
        snap = self._get_snapshot(model)
        now = time.time()
        snap.failures += 1
        snap.total += 1
        snap.last_failure_time = now
        self._events[model].append((now, False))
        self._prune_events(model, now)

        if snap.state == CircuitState.HALF_OPEN:
            snap.state = CircuitState.OPEN
            snap.opened_at = now
            snap.half_open_probes_succeeded = 0
            snap.half_open_probes_sent = 0
            return

        if snap.failures >= self.failure_threshold:
            snap.state = CircuitState.OPEN
            snap.opened_at = now
            return

        events = list(self._events.get(model, []))
        if len(events) >= 5:
            recent = [e for e in events if now - e[0] <= self.window_seconds]
            if len(recent) >= 5:
                error_rate = sum(1 for _, ok in recent if not ok) / len(recent)
                if error_rate >= self.error_rate_threshold:
                    snap.state = CircuitState.OPEN
                    snap.opened_at = now

    def _prune_events(self, model: str, now: float):
        events = self._events.get(model)
        if not events:
            return
        cutoff = now - self.window_seconds
        while events and events[0][0] < cutoff:
            events.popleft()

    def is_available(self, model: str) -> bool:
        snap = self._get_snapshot(model)
        now = time.time()
        if snap.state == CircuitState.CLOSED:
            return True
        if snap.state == CircuitState.OPEN:
            if now - snap.opened_at >= self.cooldown_seconds:
                snap.state = CircuitState.HALF_OPEN
                snap.half_open_probes_succeeded = 0
                snap.half_open_probes_sent = 0
                return True
            return False
        return True

    async def acquire_slot(self, model: str) -> bool:
        return self.is_available(model)

    async def _health_check_loop(self):
        while True:
            await asyncio.sleep(self.health_check_interval)
            if self.health_check_fn:
                try:
                    await asyncio.wait_for(
                        self.health_check_fn(),
                        timeout=self.health_check_timeout,
                    )
                    for model in self._snapshots:
                        snap = self._snapshots[model]
                        if snap.state == CircuitState.OPEN:
                            snap.state = CircuitState.HALF_OPEN
                            snap.half_open_probes_succeeded = 0
                            snap.half_open_probes_sent = 0
                except Exception:
                    pass

    def get_state(self, model: str) -> str:
        return self._get_snapshot(model).state.value
