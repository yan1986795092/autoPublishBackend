"""租户上下文管理

每个 HTTP 请求对应一个 tenant（用户），通过 ContextVar 在请求作用域内传递。
SQLAlchemy Event Hook 会读取该 ContextVar，自动注入 user_id 过滤。

使用场景：
  - get_current_user() 调用 set_tenant(user.id) 设置
  - tenant_scoped_user() 依赖在请求结束时 clear_tenant() 清理
  - 后台任务/迁移脚本使用 run_as_system() 上下文管理器临时禁用

为什么用 ContextVar 而非 threading.local？
  - FastAPI 是 async 框架，多请求并发
  - threading.local 不能跨 await 传播
  - ContextVar 是 asyncio 官方推荐的请求作用域隔离方案
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

# 当前请求所属用户 ID（None = 系统级查询）
_tenant_id: ContextVar[Optional[str]] = ContextVar("tenant_id", default=None)


def set_tenant(user_id: str) -> None:
    """设置当前请求的租户 ID（在 get_current_user 中调用）

    Args:
        user_id: 已认证用户的 UUID 字符串
    """
    _tenant_id.set(user_id)


def get_tenant() -> Optional[str]:
    """获取当前租户 ID

    Returns:
        用户 ID 字符串，如果当前在系统级上下文中则返回 None
    """
    return _tenant_id.get()


def clear_tenant() -> None:
    """清除租户上下文（请求结束时调用，防止跨请求污染）

    与 _tenant_id.set(None) 等价，但语义更清晰。
    """
    _tenant_id.set(None)


@contextmanager
def run_as_system() -> Iterator[None]:
    """系统级操作的上下文管理器（迁移脚本、后台任务、定时任务）

    用法：
        from src.core.tenant import run_as_system

        with run_as_system():
            # 这里的所有 db.query 不会被自动注入 user_id 过滤
            all_users = db.query(User).all()

    Yields:
        None
    """
    token = _tenant_id.set(None)
    try:
        yield
    finally:
        _tenant_id.reset(token)


def has_tenant() -> bool:
    """快速检查当前是否有租户上下文

    比 get_tenant() is not None 略快，因为不需要返回 token。
    """
    return _tenant_id.get() is not None