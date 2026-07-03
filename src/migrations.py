"""注册的数据库迁移。

每个迁移都是一个通过 @migration_manager.register 装饰器注册的函数。
迁移管理器会跟踪哪些迁移已在 _migrations 表中应用。
"""

from __future__ import annotations
import logging

from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)

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
                    embedding vector(1024),
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


@migration_manager.register(
    "fix_vector_search_dim",
    "修正 vector_search 表 embedding 列为 1024 维（适配 Zhipu embedding-2）",
)
def fix_vector_search_dim(engine) -> None:
    """如果 vector_search 表的 embedding 列不是 1024 维，修改为 1024"""
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "vector_search" not in inspector.get_table_names():
        return
    columns = {c["name"]: c for c in inspector.get_columns("vector_search")}
    col = columns.get("embedding", {})
    col_type = str(col.get("type", ""))
    if "1024" not in col_type:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE vector_search ALTER COLUMN embedding TYPE vector(1024)"))
            conn.commit()
            logger.info("vector_search 列改为 vector(1024)")
    else:
        logger.info("vector_search 已是 1024 维，跳过")


@migration_manager.register(
    "add_agent_tasks_conversation_id",
    "向 agent_tasks 表添加 conversation_id 列",
)
def add_agent_tasks_conversation_id(engine) -> None:
    inspector = inspect(engine)
    if "agent_tasks" not in inspector.get_table_names():
        return
    columns = [c["name"] for c in inspector.get_columns("agent_tasks")]
    if "conversation_id" not in columns:
        with engine.connect() as conn:
            conn.execute(
                text("ALTER TABLE agent_tasks ADD COLUMN conversation_id VARCHAR(36)")
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS ix_agent_tasks_conv ON agent_tasks(conversation_id)")
            )
            conn.commit()
            logger.info("Added conversation_id to agent_tasks")
    else:
        logger.info("agent_tasks already has conversation_id, skipped")


@migration_manager.register(
    "add_agent_tasks_model_name",
    "向 agent_tasks 表添加 model_name 列",
)
def add_agent_tasks_model_name(engine) -> None:
    inspector = inspect(engine)
    if "agent_tasks" not in inspector.get_table_names():
        return
    columns = [c["name"] for c in inspector.get_columns("agent_tasks")]
    if "model_name" not in columns:
        with engine.connect() as conn:
            conn.execute(
                text("ALTER TABLE agent_tasks ADD COLUMN model_name VARCHAR(100)")
            )
            conn.commit()
            logger.info("Added model_name to agent_tasks")
    else:
        logger.info("agent_tasks already has model_name, skipped")


@migration_manager.register(
    "add_contents_is_ai_generated",
    "向 contents 表添加 is_ai_generated 列并回填 AI 生成的内容",
)
def add_contents_is_ai_generated(engine) -> None:
    inspector = inspect(engine)
    if "contents" not in inspector.get_table_names():
        return
    columns = [c["name"] for c in inspector.get_columns("contents")]
    if "is_ai_generated" not in columns:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE contents ADD COLUMN is_ai_generated BOOLEAN DEFAULT FALSE"))
            conn.execute(
                text(
                    "UPDATE contents SET is_ai_generated = TRUE "
                    "WHERE metadata_json IS NOT NULL AND metadata_json != ''"
                )
            )
            conn.commit()
            logger.info("Added is_ai_generated to contents and backfilled AI content")
    else:
        logger.info("contents already has is_ai_generated, skipped")


@migration_manager.register(
    "add_conversations_is_pinned",
    "向 chat_conversations 表添加 is_pinned 列用于置顶功能",
)
def add_conversations_is_pinned(engine) -> None:
    inspector = inspect(engine)
    if "chat_conversations" not in inspector.get_table_names():
        return
    columns = [c["name"] for c in inspector.get_columns("chat_conversations")]
    if "is_pinned" not in columns:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE chat_conversations ADD COLUMN is_pinned BOOLEAN DEFAULT FALSE"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chat_conversations_pinned ON chat_conversations(is_pinned)"))
            conn.commit()
            logger.info("Added is_pinned to chat_conversations")
    else:
        logger.info("chat_conversations already has is_pinned, skipped")


# =====================================================================
# Phase 5: 系统模板/技能拆分
# 把 user_id IS NULL 的数据迁移到 system_* 表，并把 user_id 改为 NOT NULL
# =====================================================================

@migration_manager.register(
    "phase5_create_system_prompt_templates",
    "创建 system_prompt_templates 表（系统内置提示词模板）",
)
def create_system_prompt_templates(engine) -> None:
    """创建 system_prompt_templates 表"""
    inspector = inspect(engine)
    if "system_prompt_templates" in inspector.get_table_names():
        logger.info("system_prompt_templates already exists, skipped")
        return

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE system_prompt_templates (
                id VARCHAR(36) PRIMARY KEY,
                code VARCHAR(50) UNIQUE NOT NULL,
                platform_code VARCHAR(50),
                name VARCHAR(100) NOT NULL,
                description TEXT,
                system_prompt TEXT,
                user_template TEXT,
                variables_json TEXT,
                model_preferred VARCHAR(50),
                temperature FLOAT DEFAULT 0.7,
                max_tokens INTEGER DEFAULT 2000,
                is_active BOOLEAN DEFAULT TRUE,
                use_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("CREATE INDEX ix_system_prompt_templates_code ON system_prompt_templates(code)"))
        conn.execute(text("CREATE INDEX ix_system_prompt_templates_platform ON system_prompt_templates(platform_code)"))
        conn.commit()
        logger.info("Created system_prompt_templates table")


@migration_manager.register(
    "phase5_create_system_skills",
    "创建 system_skills 表（系统内置 Skill）",
)
def create_system_skills(engine) -> None:
    """创建 system_skills 表"""
    inspector = inspect(engine)
    if "system_skills" in inspector.get_table_names():
        logger.info("system_skills already exists, skipped")
        return

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE system_skills (
                id VARCHAR(36) PRIMARY KEY,
                code VARCHAR(50) UNIQUE NOT NULL,
                category VARCHAR(20) DEFAULT 'writing',
                icon VARCHAR(10) DEFAULT '✍️',
                author VARCHAR(100),
                version VARCHAR(20),
                tags VARCHAR(500),
                is_active BOOLEAN DEFAULT TRUE,
                use_count INTEGER DEFAULT 0,
                name VARCHAR(100) NOT NULL,
                description TEXT,
                system_prompt TEXT NOT NULL,
                user_prompt_template TEXT,
                preset_plan_json TEXT,
                content_types_json TEXT,
                config_schema_json TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("CREATE INDEX ix_system_skills_code ON system_skills(code)"))
        conn.execute(text("CREATE INDEX ix_system_skills_active ON system_skills(is_active)"))
        conn.commit()
        logger.info("Created system_skills table")


@migration_manager.register(
    "phase5_migrate_prompt_templates",
    "把 user_id IS NULL 的 prompt_templates 迁移到 system_prompt_templates",
)
def migrate_prompt_templates_null_to_system(engine) -> None:
    """迁移 prompt_templates.user_id IS NULL 的记录到 system_prompt_templates"""
    inspector = inspect(engine)
    if "prompt_templates" not in inspector.get_table_names():
        return
    if "system_prompt_templates" not in inspector.get_table_names():
        logger.warning("system_prompt_templates not created yet, skipping migration")
        return

    with engine.connect() as conn:
        # 查出有多少 NULL user_id
        null_count = conn.execute(text(
            "SELECT COUNT(*) FROM prompt_templates WHERE user_id IS NULL"
        )).scalar()

        if null_count and null_count > 0:
            # 用 name 作为 code（如果重复则加后缀）
            conn.execute(text("""
                INSERT INTO system_prompt_templates (
                    id, code, platform_code, name, description,
                    system_prompt, user_template, variables_json,
                    model_preferred, temperature, max_tokens,
                    is_active, use_count, created_at, updated_at
                )
                SELECT
                    id, 'migrated_' || SUBSTRING(id, 1, 8), platform_code, name, description,
                    system_prompt, user_template, variables_json,
                    model_preferred, temperature, max_tokens,
                    TRUE, use_count, created_at, updated_at
                FROM prompt_templates
                WHERE user_id IS NULL
            """))
            # 删除原记录
            conn.execute(text("DELETE FROM prompt_templates WHERE user_id IS NULL"))
            logger.info(f"Migrated {null_count} prompt templates to system_prompt_templates")
        else:
            logger.info("No prompt_templates with NULL user_id, skipping migration")

        conn.commit()


@migration_manager.register(
    "phase5_migrate_skills",
    "把 user_id IS NULL 的 skills 迁移到 system_skills",
)
def migrate_skills_null_to_system(engine) -> None:
    """迁移 skills.user_id IS NULL 的记录到 system_skills"""
    inspector = inspect(engine)
    if "skills" not in inspector.get_table_names():
        return
    if "system_skills" not in inspector.get_table_names():
        logger.warning("system_skills not created yet, skipping migration")
        return

    with engine.connect() as conn:
        null_count = conn.execute(text(
            "SELECT COUNT(*) FROM skills WHERE user_id IS NULL"
        )).scalar()

        if null_count and null_count > 0:
            conn.execute(text("""
                INSERT INTO system_skills (
                    id, code, category, icon, author, version, tags,
                    is_active, use_count,
                    name, description, system_prompt, user_prompt_template,
                    preset_plan_json, content_types_json, config_schema_json,
                    created_at, updated_at
                )
                SELECT
                    id, 'migrated_' || SUBSTRING(id, 1, 8), category, icon, author, version, tags,
                    is_active, use_count,
                    name, description, system_prompt, user_prompt_template,
                    preset_plan_json, content_types_json, config_schema_json,
                    created_at, updated_at
                FROM skills
                WHERE user_id IS NULL
            """))
            conn.execute(text("DELETE FROM skills WHERE user_id IS NULL"))
            logger.info(f"Migrated {null_count} skills to system_skills")
        else:
            logger.info("No skills with NULL user_id, skipping migration")

        conn.commit()


@migration_manager.register(
    "phase5_set_user_id_not_null",
    "把 prompt_templates.user_id 和 skills.user_id 改为 NOT NULL",
)
def set_user_id_not_null(engine) -> None:
    """Phase 5 收尾：把 user_id 改为 NOT NULL

    注意：
      - 必须先跑 phase5_migrate_* 迁移 NULL 数据
      - 此迁移后所有新插入必须提供 user_id
      - ORM hook 的 before_flush 会校验
    """
    inspector = inspect(engine)
    with engine.connect() as conn:
        # PostgreSQL: ALTER COLUMN ... SET NOT NULL
        # SQLite: 不支持 SET NOT NULL（需重建表），跳过 SQLite 场景
        if engine.dialect.name == "postgresql":
            for table in ("prompt_templates", "skills"):
                if table not in inspector.get_table_names():
                    continue
                # 先确认没有 NULL
                null_count = conn.execute(text(
                    f"SELECT COUNT(*) FROM {table} WHERE user_id IS NULL"
                )).scalar()
                if null_count and null_count > 0:
                    raise RuntimeError(
                        f"[Phase 5] {table} 仍有 {null_count} 条 NULL user_id，"
                        f"请先运行 phase5_migrate_* 迁移"
                    )
                conn.execute(text(
                    f"ALTER TABLE {table} ALTER COLUMN user_id SET NOT NULL"
                ))
                logger.info(f"{table}.user_id -> NOT NULL")
            conn.commit()
        else:
            logger.info(f"Skipping NOT NULL constraint on {engine.dialect.name} (Phase 5 仅 PostgreSQL)")


# =====================================================================
# Phase 1: Parts 模型重构
# =====================================================================

@migration_manager.register(
    "chat_message_parts",
    "创建 chat_message_parts 表（Part 持久化，一对多关联 chat_message_records）",
)
def create_chat_message_parts(engine) -> None:
    """创建 chat_message_parts 表（spec §3.2.1）

    表结构：
      - id           VARCHAR(36)  PRIMARY KEY
      - message_id   VARCHAR(36)  NOT NULL  FK → chat_message_records.id
      - seq          INTEGER      NOT NULL   （同一 message 内的顺序）
      - type         VARCHAR(20)  NOT NULL   （text/reasoning/tool_call/artifact/plan/error）
      - state        VARCHAR(20)  NOT NULL   DEFAULT 'streaming'
      - payload_json TEXT         NOT NULL   （各 type 的 payload，JSON）
      - server_id    VARCHAR(64)  可选       （服务端分配，用于幂等去重）
      - created_at   TIMESTAMP WITH TIME ZONE
      - completed_at TIMESTAMP WITH TIME ZONE
    """
    inspector = inspect(engine)

    if "chat_message_parts" in inspector.get_table_names():
        logger.info("chat_message_parts 已存在，跳过")
        return

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE chat_message_parts (
                id              VARCHAR(36) PRIMARY KEY,
                message_id      VARCHAR(36) NOT NULL,
                seq             INTEGER NOT NULL,
                type            VARCHAR(20) NOT NULL,
                state           VARCHAR(20) NOT NULL DEFAULT 'streaming',
                payload_json    TEXT NOT NULL DEFAULT '{}',
                server_id       VARCHAR(64),
                created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                completed_at    TIMESTAMP WITH TIME ZONE
            )
        """))
        conn.execute(text(
            "ALTER TABLE chat_message_parts "
            "ADD CONSTRAINT fk_parts_message "
            "FOREIGN KEY (message_id) REFERENCES chat_message_records(id) ON DELETE CASCADE"
        ))
        conn.execute(text(
            "CREATE INDEX ix_parts_message_seq "
            "ON chat_message_parts(message_id, seq)"
        ))
        conn.execute(text(
            "CREATE UNIQUE INDEX ix_parts_message_server "
            "ON chat_message_parts(message_id, server_id) "
            "WHERE server_id IS NOT NULL"
        ))
        conn.commit()

    logger.info("chat_message_parts 表创建完成")
