"""Sink 协议 — Phase 1 引入。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..log_record import LogRecord


@runtime_checkable
class Sink(Protocol):
    """日志输出端协议。

    write() 可以抛异常（Dispatcher 捕获并入 metrics，不阻塞业务），
    不要静默吞错（设计原则 P6）。
    """

    name: str

    def write(self, record: LogRecord) -> None:
        ...
