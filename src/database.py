from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
import structlog
import time
from src.config import get_settings

logger = structlog.get_logger("db")

settings = get_settings()

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    poolclass=StaticPool if "sqlite" in settings.database_url else None,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── SQLAlchemy 查询日志 ──────────────────────────────────────────

def _should_log_db() -> bool:
    """检查数据库日志是否启用（由环境变量控制）。"""
    import os
    return os.getenv("ENABLE_DB_LOG", "false").lower() in ("true", "1", "yes")


@event.listens_for(Engine, "before_cursor_execute")
def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    if not _should_log_db():
        return
    conn.info.setdefault("_db_query_start", {}).setdefault(id(cursor), time.perf_counter())


@event.listens_for(Engine, "after_cursor_execute")
def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    if not _should_log_db():
        return
    start_times = conn.info.get("_db_query_start", {})
    start = start_times.pop(id(cursor), None)
    if start is None:
        return
    duration_ms = (time.perf_counter() - start) * 1000

    # 清理 SQL：合并空白字符，截断至 200 字符
    sql = " ".join(statement.split())
    if len(sql) > 200:
        sql = sql[:197] + "..."

    row_count = cursor.rowcount if hasattr(cursor, "rowcount") and cursor.rowcount >= 0 else None

    logger.debug(
        sql,
        _type="db",
        statement=sql,
        duration_ms=round(duration_ms, 2),
        rows=row_count,
        engine=engine.url.drivername,
    )


def setup_db_logging(enabled: bool = False) -> None:
    """在运行时启用或禁用 SQLAlchemy 查询日志记录。"""
    import os
    os.environ["ENABLE_DB_LOG"] = "true" if enabled else "false"


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
