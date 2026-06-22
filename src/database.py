from typing import Generator
import time

import structlog
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from src.config import get_settings

logger = structlog.get_logger("db")

settings = get_settings()

_ENGINE_KWARGS = {
    "pool_pre_ping": True,
    "echo": False,
}

if settings.database_url.startswith("sqlite"):
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        **_ENGINE_KWARGS,
    )
else:
    engine = create_engine(
        settings.database_url,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=10,
        pool_recycle=3600,
        pool_use_lifo=True,
        **_ENGINE_KWARGS,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _should_log_db() -> bool:
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
    import os

    os.environ["ENABLE_DB_LOG"] = "true" if enabled else "false"


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
