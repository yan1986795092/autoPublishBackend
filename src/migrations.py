"""Registered database migrations.

Each migration is a function registered with the @migration_manager.register decorator.
The migration manager tracks which have been applied in the _migrations table.
"""

from __future__ import annotations

from sqlalchemy import inspect, text

from src.core.migrations import migration_manager


@migration_manager.register("add_category_column", "Add category column to platforms table")
def add_category_column(engine) -> None:
    inspector = inspect(engine)
    columns = [c["name"] for c in inspector.get_columns("platforms")]
    if "category" not in columns:
        with engine.connect() as conn:
            conn.execute(
                text(
                    "ALTER TABLE platforms "
                    "ADD COLUMN category VARCHAR(50) DEFAULT 'social_media'"
                )
            )
            conn.commit()
