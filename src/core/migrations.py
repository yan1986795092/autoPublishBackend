"""通用数据库迁移框架。

用法：
    from src.core.migrations import migration_manager

    @migration_manager.register("add_category_column", "向 platforms 表添加 category 列")
    def add_category_column(engine):
        from sqlalchemy import inspect, text
        inspector = inspect(engine)
        if 'category' not in [c['name'] for c in inspector.get_columns('platforms')]:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE platforms ADD COLUMN category VARCHAR(50) DEFAULT 'social_media'"))
                conn.commit()

    # 在应用生命周期中：
    migration_manager.run_all(engine)
"""

from __future__ import annotations

from typing import Callable

import structlog

logger = structlog.get_logger(__name__)


class Migration:
    """单个迁移"""

    def __init__(self, name: str, description: str, fn: Callable) -> None:
        self.name = name
        self.description = description
        self.fn = fn


class MigrationManager:
    """管理注册的迁移并跟踪哪些已应用"""

    def __init__(self):
        self._migrations: list[Migration] = []

    def register(self, name: str, description: str = "") -> Callable:
        """装饰器：注册一个迁移函数

        Args:
            name: 迁移的唯一名称（用于幂等追踪）
            description: 迁移描述
        """

        def decorator(fn: Callable) -> Callable:
            self._migrations.append(Migration(name, description or fn.__name__, fn))
            return fn

        return decorator

    def run_all(self, engine) -> int:
        """执行所有未应用的迁移

        Args:
            engine: SQLAlchemy Engine 实例

        Returns:
            成功执行的迁移数量
        """
        self._ensure_table(engine)
        applied = self._get_applied(engine)

        count = 0
        for m in self._migrations:
            if m.name in applied:
                continue
            logger.info("Running migration: %s — %s", m.name, m.description,
                        _type="system", event_name="migration_run", message=f"Running migration: {m.name}", status="running")
            try:
                m.fn(engine)
                self._mark_applied(engine, m.name)
                count += 1
                logger.info("Migration %s applied successfully", m.name,
                            _type="system", event_name="migration_done", message=f"Migration {m.name} applied", status="ok")
            except Exception as e:
                logger.error("Migration %s failed: %s", m.name, e,
                             _type="system", event_name="migration_error", message=f"Migration {m.name} failed: {e}", status="error")
                raise

        if count == 0:
            logger.info("No pending migrations",
                        _type="system", event_name="migration_skip", message="No pending migrations", status="ok")

        return count

    def _ensure_table(self, engine) -> None:
        from sqlalchemy import inspect, text

        inspector = inspect(engine)
        if "_migrations" not in inspector.get_table_names():
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "CREATE TABLE _migrations ("
                        "  name VARCHAR PRIMARY KEY,"
                        "  applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                        ")"
                    )
                )
                conn.commit()

    def _get_applied(self, engine) -> set[str]:
        from sqlalchemy import text

        with engine.connect() as conn:
            rows = conn.execute(text("SELECT name FROM _migrations")).fetchall()
            return {row[0] for row in rows}

    def _mark_applied(self, engine, name: str) -> None:
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(
                text("INSERT INTO _migrations (name) VALUES (:name)"),
                {"name": name},
            )
            conn.commit()

    def list_pending(self, engine) -> list[Migration]:
        """列出所有尚未应用的迁移"""
        self._ensure_table(engine)
        applied = self._get_applied(engine)
        return [m for m in self._migrations if m.name not in applied]


# 全局单例
migration_manager = MigrationManager()
