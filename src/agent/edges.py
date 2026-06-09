"""条件边路由函数 — 集中管理所有 conditional edges

简化后只有一个条件边路由：should_continue_or_finish，
定义在 nodes.py 中，此处仅为向后兼容导出。
"""
from __future__ import annotations

from .nodes import should_continue_or_finish

__all__ = ["should_continue_or_finish"]
