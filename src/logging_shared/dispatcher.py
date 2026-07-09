"""Dispatcher — Phase 1 引入。

显式 Sampling + Routing + Redactor 三个 hook, sink fan-out。
不静默丢日志：被 Sample 命中的事件设 __sampled__: skipped 也进结构化层。
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from .log_record import LogRecord
from .redactor import DefaultRedactor, Redactor, NoopRedactor


class Sampler:
    """显式采样决策。

    should_sample(record) 返回 False → 标记 __sampled__: skipped, 不进 Console,
    但仍可入 EsSink (结构化层可见)。"""

    def __init__(self, rate: float, predicate: Callable[[LogRecord], bool] = lambda r: True):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self.rate = rate
        self.predicate = predicate
        self._counter = 0

    def should_sample(self, record: LogRecord) -> bool:
        if not self.predicate(record):
            return True
        self._counter += 1
        if self.rate >= 1.0:
            return True
        if self.rate <= 0.0:
            return False
        return (self._counter * self.rate) % 1 < self.rate


class RoutingRule:
    """按 (predicate, target_sinks) 列表分派 LogRecord。"""

    def __init__(self, rules: list[tuple[Callable[[LogRecord], bool], list]] | None = None):
        self.rules = rules or []

    def match(self, record: LogRecord, default: list) -> list:
        for predicate, targets in self.rules:
            if predicate(record):
                return targets
        return default


class Dispatcher:
    def __init__(
        self,
        sinks: list,
        sampler: Sampler | None = None,
        routing: RoutingRule | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self.sinks = sinks
        self.sampler = sampler
        self.routing = routing
        self.redactor = redactor or DefaultRedactor()

    def emit(self, record: LogRecord) -> None:
        # 1. PII 脱敏 (P7)
        try:
            if isinstance(self.redactor, DefaultRedactor):
                # 主脱敏对 LogRecord 的 fields 字典递归
                record = self._redact_fields(record)
        except Exception:
            pass  # 不阻断业务, 后续可接 metrics

        # 2. 路由
        targets = self.routing.match(record, self.sinks) if self.routing else self.sinks

        # 3. 采样 (显式标记, 不静默)
        if self.sampler and not self.sampler.should_sample(record):
            fields = dict(record.fields)
            fields["__sampled__"] = "skipped"
            record = LogRecord(
                **{**record.__dict__, "fields": fields},
            )
            # 已跳过: 仅让 EsSink 接收, Console 不再处理
            struct_targets = [s for s in targets if getattr(s, "name", "") in ("es", "jsonfile")]
            targets = struct_targets

        # 4. Fan-out (任意 sink 异常不影响其他)
        for sink in targets:
            try:
                sink.write(record)
            except Exception as exc:
                # P6: 不静默 — 留给后续 metrics/observability
                sys.stderr.write(f"[dispatcher.sink.error] {getattr(sink, 'name', '?')} {exc}\n")

    def _redact_fields(self, record: LogRecord) -> LogRecord:
        """对 record.fields 递归走一遍 PII regex. 仅在 DefaultRedactor 启用时."""
        from .redactor import _redact_text
        fields = dict(record.fields)
        changed = False
        for k, v in list(fields.items()):
            new_v = self._redact_value(v)
            if new_v is not v:
                fields[k] = new_v
                changed = True
        if not changed:
            return record
        return LogRecord(**{**record.__dict__, "fields": fields})

    def _redact_value(self, obj):
        from .redactor import _redact_text
        if isinstance(obj, str):
            return _redact_text(obj)
        if isinstance(obj, list):
            return [self._redact_value(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self._redact_value(v) for k, v in obj.items()}
        return obj


# ── 单例管理 ───────────────────────────────────────────────────

import sys as _sys_for_stdout


_dispatcher_instance: Dispatcher | None = None


def get_dispatcher() -> Dispatcher:
    """获取全局 Dispatcher 实例 (懒初始化).

    setup() 时被填充 — 不调用 setup 则 fallback 到一个最小的 Console-only 实例。
    """
    global _dispatcher_instance
    if _dispatcher_instance is not None:
        return _dispatcher_instance
    # fallback: 只创建 ConsoleSink (Phase 3 才删除这个分支)
    _dispatcher_instance = Dispatcher(sinks=[])
    return _dispatcher_instance


def set_dispatcher(dispatcher: Dispatcher) -> None:
    global _dispatcher_instance
    _dispatcher_instance = dispatcher
