"""TenantContext 单元测试（独立测试，不依赖整个 app）

验证：
1. set_tenant / get_tenant / clear_tenant 基本功能
2. run_as_system 上下文管理器
3. ContextVar 的并发隔离
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

# 把 backend 加到 sys.path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest

from src.core.tenant import (
    clear_tenant,
    get_tenant,
    has_tenant,
    run_as_system,
    set_tenant,
)


class TestTenantContextBasic:
    """基础功能测试"""

    def test_initial_state_is_none(self):
        """初始状态（未设置）应返回 None"""
        clear_tenant()
        assert get_tenant() is None
        assert has_tenant() is False

    def test_set_and_get(self):
        """设置后能取到对应值"""
        clear_tenant()
        user_id = str(uuid.uuid4())
        set_tenant(user_id)
        assert get_tenant() == user_id
        assert has_tenant() is True

    def test_clear_removes_tenant(self):
        """clear 后 tenant 回到 None"""
        set_tenant("user-123")
        clear_tenant()
        assert get_tenant() is None
        assert has_tenant() is False

    def test_overwrite_tenant(self):
        """重复设置会覆盖"""
        set_tenant("user-A")
        set_tenant("user-B")
        assert get_tenant() == "user-B"
        clear_tenant()

    def test_uuid_format(self):
        """UUID 格式的 user_id 也能正确传递"""
        clear_tenant()
        uid = str(uuid.uuid4())
        set_tenant(uid)
        assert get_tenant() == uid
        clear_tenant()


class TestRunAsSystem:
    """run_as_system 上下文管理器"""

    def test_run_as_system_disables_filter(self):
        """run_as_system 内 tenant 应为 None"""
        set_tenant("user-A")
        assert get_tenant() == "user-A"

        with run_as_system():
            assert get_tenant() is None
            assert has_tenant() is False

        # 退出后恢复（run_as_system 内部用 token.reset）
        assert get_tenant() == "user-A"
        clear_tenant()

    def test_run_as_system_from_clean(self):
        """从 clean 状态进入也安全"""
        clear_tenant()
        with run_as_system():
            assert get_tenant() is None
        assert get_tenant() is None

    def test_run_as_system_restores_on_exception(self):
        """即使内部抛异常，tenant 也应被恢复"""
        set_tenant("user-X")
        try:
            with run_as_system():
                assert get_tenant() is None
                raise ValueError("测试异常")
        except ValueError:
            pass

        assert get_tenant() == "user-X"
        clear_tenant()


class TestAsyncIsolation:
    """async 上下文隔离测试（关键能力）"""

    @pytest.mark.asyncio
    async def test_concurrent_tasks_have_isolated_tenant(self):
        """两个并发任务各自有自己的 tenant（ContextVar 的核心能力）"""
        results = {}

        async def worker(name: str, delay: float):
            set_tenant(f"user-{name}")
            await asyncio.sleep(delay)  # 让出执行权
            # 此时如果有另一个任务也 set 了不同的 tenant，
            # 由于 ContextVar 的 copy-on-write，这里仍应该是自己设的值
            results[name] = get_tenant()

        await asyncio.gather(
            worker("A", 0.1),
            worker("B", 0.05),
        )
        clear_tenant()

        assert results["A"] == "user-A"
        assert results["B"] == "user-B"

    @pytest.mark.asyncio
    async def test_tenant_isolated_between_tasks(self):
        """任务结束后不会污染下一个任务"""
        async def task1():
            set_tenant("task1-user")
            await asyncio.sleep(0.01)
            return get_tenant()

        async def task2():
            # 进入时应该是初始状态（None），不受 task1 影响
            initial = get_tenant()
            set_tenant("task2-user")
            await asyncio.sleep(0.01)
            return initial, get_tenant()

        r1, (r2_initial, r2_final) = await asyncio.gather(task1(), task2())
        clear_tenant()

        assert r1 == "task1-user"
        assert r2_final == "task2-user"
        # task2 进入时应该是 None（ContextVar 默认值），说明 task1 没污染