#!/usr/bin/env python3
"""Schema diff 工具 — 对比数据库实际结构与 SQLAlchemy 模型定义。

用法:
    # 本地
    python scripts/diff_schema.py postgresql+psycopg2://autopublish:autopublish_dev@localhost:5432/autopublish

    # 生产
    python scripts/diff_schema.py postgresql+psycopg2://autopublish:<PASS>@<PROD_HOST>:5432/autopublish

输出：
    - MISSING TABLES:  当前 DB 缺少的表
    - MISSING COLUMNS: 表存在但列缺失
    - MISSING INDEXES:  缺少的索引（不含主键/唯一自动创建的索引）
    - MIGRATION STATUS: _migrations 表中的已应用迁移 vs 代码中注册的迁移
"""
from __future__ import annotations

import sys
from pathlib import Path

# 将项目根目录加入 path，以便 import src.models
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

# 触发迁移注册
import src.migrations  # noqa: F401 — 注册迁移

# 导入模型 Base（会触发所有模型类定义）
from src.models import Base  # noqa: F401
from src.core.migrations import migration_manager


def get_inspector(engine: Engine):
    return inspect(engine)


def list_tables(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def list_columns(engine: Engine, table: str) -> set[str]:
    try:
        return {c["name"] for c in inspect(engine).get_columns(table)}
    except Exception:
        return set()


def list_indexes(engine: Engine, table: str) -> set[str]:
    try:
        return {
            idx["name"]
            for idx in inspect(engine).get_indexes(table)
            if idx["name"] and not idx["name"].startswith("ix_") and not idx.get("unique", False)
        }
    except Exception:
        return set()


def get_model_tables() -> set[str]:
    """从 Base.metadata 提取所有表名"""
    return {table.name for table in Base.metadata.tables.values()}


def get_model_columns(table_name: str) -> set[tuple[str, str]]:
    """返回 (列名, 列类型字符串) 的集合，用于精细比较"""
    table = Base.metadata.tables.get(table_name)
    if table is None:
        return set()
    result = set()
    for col in table.columns:
        col_type = str(col.type).lower()
        result.add((col.name, col_type))
    return result


def get_model_indexes(table_name: str) -> set[str]:
    table = Base.metadata.tables.get(table_name)
    if table is None:
        return set()
    result = set()
    for idx in table.indexes:
        result.add(idx.name)
    return result


def print_diff(engine: Engine, label: str = "TARGET DB"):
    print(f"\n{'='*60}")
    print(f"  Schema Diff Report — {label}")
    print(f"{'='*60}\n")

    insp = inspect(engine)
    actual_tables = list_tables(engine)
    model_tables = get_model_tables()

    missing_tables = model_tables - actual_tables
    if missing_tables:
        print(f"[!] MISSING TABLES ({len(missing_tables)}):")
        for t in sorted(missing_tables):
            print(f"    - {t}")
    else:
        print("[OK] All expected tables exist.")

    # 列级别 diff
    missing_cols = []
    extra_cols = []
    type_mismatches = []
    for table in sorted(model_tables & actual_tables):
        actual_col_set = list_columns(engine, table)
        model_col_tuples = get_model_columns(table)
        model_col_names = {c[0] for c in model_col_tuples}

        for col_name, col_type in sorted(model_col_tuples):
            if col_name not in actual_col_set:
                missing_cols.append((table, col_name, col_type))

        # 检查额外列（DB 有但模型没有）— 通常是遗留，提示但不报错
        for col_name in sorted(actual_col_set - model_col_names):
            extra_cols.append((table, col_name))

    if missing_cols:
        print(f"\n[!] MISSING COLUMNS ({len(missing_cols)}):")
        for tbl, col, ctype in missing_cols:
            print(f"    - {tbl}.{col} ({ctype})")
    else:
        print("\n[OK] All expected columns present.")

    if extra_cols:
        print(f"\n[i] EXTRA COLUMNS in DB (not in model, may be legacy):")
        for tbl, col in extra_cols[:20]:
            print(f"    - {tbl}.{col}")
        if len(extra_cols) > 20:
            print(f"    ... and {len(extra_cols) - 20} more")

    # 索引 diff（仅关注显式定义的索引名）
    missing_idx = []
    for table in sorted(model_tables & actual_tables):
        actual_idx = list_indexes(engine, table)
        model_idx_names = get_model_indexes(table)
        for idx_name in model_idx_names:
            if idx_name not in actual_idx:
                missing_idx.append((table, idx_name))

    if missing_idx:
        print(f"\n[!] MISSING INDEXES ({len(missing_idx)}):")
        for tbl, idx in missing_idx:
            print(f"    - {tbl}.{idx}")
    else:
        print("\n[OK] All expected indexes present.")

    # Migration 状态
    print("\n" + "-"*40)
    print("Migration Status:")
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT name, applied_at FROM _migrations ORDER BY applied_at")).fetchall()
        applied = {r[0] for r in rows}
        print(f"  Applied: {len(applied)}")
        for name, at in rows:
            print(f"    {name}  ({at})")

        all_registered = {m.name for m in migration_manager._migrations}
        pending = all_registered - applied
        if pending:
            print(f"\n[!] PENDING MIGRATIONS ({len(pending)}):")
            for name in sorted(pending):
                m = next(m for m in migration_manager._migrations if m.name == name)
                print(f"    - {name} -- {m.description}")
        else:
            print("\n[OK] All migrations applied.")
    except Exception as e:
        print(f"  (Could not read _migrations: {e})")

    print("\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python diff_schema.py <DATABASE_URL>")
        print("Example:")
        print("  python diff_schema.py postgresql+psycopg2://autopublish:secret@prod-host:5432/autopublish")
        sys.exit(1)

    db_url = sys.argv[1]
    engine = create_engine(db_url, echo=False)

    try:
        print_diff(engine, label=db_url)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
