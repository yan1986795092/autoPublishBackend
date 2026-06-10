"""Metrics stubs — OTel metrics removed.

The original OTel InMemoryMetricReader was never actually exported.
These stubs keep backward compatibility for existing importers.
All functions are no-ops.
"""

from __future__ import annotations


def create_meters() -> None:
    """Stub: no-op. Metrics removed."""
    pass


def record_agent_task(agent_type: str = "", task_type: str = "", status: str = "", duration_s: float = 0) -> None:
    """Stub: no-op."""
    pass


def record_agent_run(status: str = "", skill_name: str = "", duration_s: float = 0) -> None:
    """Stub: no-op."""
    pass


agent_cache_hit = None
agent_cache_miss = None
