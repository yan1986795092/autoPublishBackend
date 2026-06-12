"""注册的数据库迁移。

每个迁移都是一个通过 @migration_manager.register 装饰器注册的函数。
迁移管理器会跟踪哪些迁移已在 _migrations 表中应用。
"""

from __future__ import annotations

from sqlalchemy import inspect, text

from src.core.migrations import migration_manager


@migration_manager.register("add_category_column", "向 platforms 表添加 category 列")
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


@migration_manager.register(
    "rename_thinking_trace_json_to_agent_events_json",
    "添加 agent_events_json 列并从 thinking_trace_json 回填",
)
def rename_thinking_trace_json_to_agent_events_json(engine) -> None:
    inspector = inspect(engine)
    if "agent_tasks" not in inspector.get_table_names():
        return

    columns = [c["name"] for c in inspector.get_columns("agent_tasks")]
    if "agent_events_json" not in columns:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE agent_tasks ADD COLUMN agent_events_json TEXT"))
            conn.commit()

    columns = [c["name"] for c in inspector.get_columns("agent_tasks")]
    if "thinking_trace_json" in columns:
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE agent_tasks SET agent_events_json = thinking_trace_json "
                "WHERE agent_events_json IS NULL AND thinking_trace_json IS NOT NULL"
            ))
            conn.commit()


@migration_manager.register(
    "create_vector_search_extension",
    "启用 pgvector 扩展 + 创建 vector_search 表",
)
def create_vector_search_extension(engine) -> None:
    inspector = inspect(engine)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    if "vector_search" not in inspector.get_table_names():
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE vector_search (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    embedding vector(1536),
                    namespace VARCHAR(50) DEFAULT 'general',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))
            conn.execute(text("""
                CREATE INDEX idx_vector_search_embedding
                ON vector_search
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """))
            conn.execute(text("""
                CREATE INDEX idx_vector_search_namespace
                ON vector_search (namespace)
            """))
            conn.commit()
