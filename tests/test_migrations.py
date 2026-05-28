"""Tests for the MigrationManager framework."""

import os
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool

from src.core.migrations import MigrationManager


@pytest.fixture
def engine():
    """Create a fresh in-memory SQLite engine for each test."""
    e = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    yield e


@pytest.fixture
def mgr():
    """Fresh MigrationManager for each test."""
    return MigrationManager()


class TestMigrationManager:
    """MigrationManager 单元测试"""

    def test_register_decorator(self, mgr):
        @mgr.register("test_mig", "A test migration")
        def my_mig(engine):
            pass

        assert len(mgr._migrations) == 1
        assert mgr._migrations[0].name == "test_mig"
        assert mgr._migrations[0].description == "A test migration"

    def test_register_default_description(self, mgr):
        """未提供 description 时使用函数名"""
        @mgr.register("test_mig")
        def my_func(engine):
            pass

        assert mgr._migrations[0].description == "my_func"

    def test_run_all_runs_pending(self, mgr, engine):
        ran = []

        @mgr.register("first", "")
        def first(e):
            ran.append("first")

        @mgr.register("second", "")
        def second(e):
            ran.append("second")

        count = mgr.run_all(engine)
        assert count == 2
        assert ran == ["first", "second"]

    def test_run_all_is_idempotent(self, mgr, engine):
        ran = []

        @mgr.register("only_one", "")
        def only_one(e):
            ran.append("executed")

        mgr.run_all(engine)
        assert len(ran) == 1

        mgr.run_all(engine)
        assert len(ran) == 1  # still 1 — didn't re-run

    def test_creates_tracking_table(self, mgr, engine):
        @mgr.register("a", "")
        def a(e):
            pass

        mgr.run_all(engine)
        inspector = inspect(engine)
        assert "_migrations" in inspector.get_table_names()

        with engine.connect() as conn:
            rows = conn.execute(text("SELECT name FROM _migrations")).fetchall()
            names = [r[0] for r in rows]
        assert "a" in names

    def test_migration_applied_only_once(self, mgr, engine):
        counter = [0]

        @mgr.register("counted", "")
        def counted(e):
            counter[0] += 1

        mgr.run_all(engine)
        mgr.run_all(engine)
        mgr.run_all(engine)
        assert counter[0] == 1  # called only once

    def test_list_pending(self, mgr, engine):
        @mgr.register("done", "")
        def done(e):
            pass

        mgr.run_all(engine)  # "done" is now applied

        @mgr.register("pending", "")
        def pending(e):
            pass

        pending_list = mgr.list_pending(engine)
        names = [m.name for m in pending_list]
        assert "done" not in names
        assert "pending" in names
        assert len(pending_list) == 1

    def test_migration_failure_stops_and_raises(self, mgr, engine):
        @mgr.register("good", "")
        def good(e):
            pass

        mgr.run_all(engine)  # "good" runs and is applied

        @mgr.register("bad", "")
        def bad(e):
            raise RuntimeError("migration failed")

        @mgr.register("after_bad", "")
        def after_bad(e):
            pass

        with pytest.raises(RuntimeError, match="migration failed"):
            mgr.run_all(engine)  # "bad" runs and raises

        # "after_bad" should NOT have been applied
        pending = mgr.list_pending(engine)
        assert any(m.name == "after_bad" for m in pending)

    def test_migration_actually_modifies_db(self, mgr, engine):
        """Integration-style: migration creates a real table"""

        @mgr.register("create_some_table", "Create a test table")
        def create_some_table(e):
            with e.connect() as conn:
                conn.execute(text("CREATE TABLE some_table (id INTEGER)"))
                conn.commit()

        mgr.run_all(engine)
        inspector = inspect(engine)
        assert "some_table" in inspector.get_table_names()

    def test_no_pending_migrations(self, mgr, engine):
        """run_all when nothing registered returns 0"""
        count = mgr.run_all(engine)
        assert count == 0
