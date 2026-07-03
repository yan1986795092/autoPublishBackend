"""一次性迁移脚本：将旧 result_json 历史数据迁移到 chat_message_parts 表。

运行方式：
    python -m src.scripts.migrate_parts_history [--batch-size 500] [--dry-run]

- 默认 dry-run（只打印数量和前 3 条样例）
- 去掉 --dry-run 才真正写入
- 已迁移的消息（有 migration_record）会跳过
- 支持断点重跑

Exit codes:
    0  无事可做或迁移完成
    1  参数错误
    2  数据库错误
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _get_connection_url(env_var: str = "DATABASE_URL") -> str:
    url = "postgresql://postgres:postgres@localhost:5432/autogitub"
    return url


def _build_migrate_parts(
    legacy_result_json: str | None,
) -> list[dict]:
    """migrate_legacy_result_to_parts 的镜像实现（不依赖 FastAPI app）。"""
    parts: list[dict] = []
    if not legacy_result_json:
        return parts
    try:
        result = json.loads(legacy_result_json)
    except (json.JSONDecodeError, TypeError):
        return parts
    if not isinstance(result, dict):
        return parts

    # 1. reasoning
    reasoning_chunks = result.get("_reasoningChunks", {})
    if isinstance(reasoning_chunks, dict):
        for key, chunks in reasoning_chunks.items():
            if not isinstance(chunks, list):
                continue
            content = "".join(c.get("content", "") for c in chunks if isinstance(c, dict))
            if not content:
                continue
            agent = key.split("|")[0] if "|" in key else key
            round_str = key.split("|")[1] if "|" in key else "0"
            try:
                round_num = int(round_str)
            except ValueError:
                round_num = 0
            parts.append({
                "type": "reasoning",
                "state": "done",
                "content": content,
                "agent": agent,
                "round": round_num,
                "collapsed_by_default": True,
            })

    # 2. tool_call
    stages = result.get("_stages", [])
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            parts.append({
                "type": "tool_call",
                "state": "done",
                "tool_name": stage.get("agent", ""),
                "display_name": stage.get("label", stage.get("agent", "")),
                "icon": stage.get("icon", "⚙️"),
                "input": {},
                "output": stage.get("output", {}),
                "duration_ms": int(float(stage.get("duration", 0)) * 1000),
                "round": stage.get("reviewRound", 0),
            })

    # 3. artifact
    output = result.get("output", {})
    if isinstance(output, dict) and output:
        body = output.get("full_text") or output.get("content") or ""
        if body:
            platform = output.get("platform", "")
            if platform in ("xiaohongshu", "xiaohongshu_native"):
                artifact_type = "social_post"
            else:
                ct = output.get("content_type", "")
                if ct == "code":
                    artifact_type = "code"
                elif ct in ("image", "img"):
                    artifact_type = "image"
                elif ct in ("tutorial", "howto"):
                    artifact_type = "tutorial"
                else:
                    artifact_type = "blog"
            parts.append({
                "type": "artifact",
                "state": "done",
                "artifact_type": artifact_type,
                "body": body,
                "title": output.get("title"),
                "platform": platform,
                "structured": output.get("structured"),
            })

    # 4. plan
    plan_json_str = result.get("plan_json") or result.get("_plan")
    if plan_json_str:
        try:
            plan_data = json.loads(plan_json_str) if isinstance(plan_json_str, str) else plan_json_str
            if isinstance(plan_data, dict) and plan_data.get("tasks"):
                parts.append({
                    "type": "plan",
                    "state": "done",
                    "goal": plan_data.get("goal", ""),
                    "reasoning": plan_data.get("reasoning", ""),
                    "tasks": plan_data.get("tasks", []),
                })
        except (json.JSONDecodeError, TypeError):
            pass

    return parts


def iter_messages_needing_migration(
    session: Session,
    batch_size: int = 500,
) -> Generator[tuple, None, None]:
    """逐批查询需要迁移的消息（assistant 角色 + 有 result_json + 未迁移）。"""
    offset = 0
    while True:
        rows = session.execute(
            text("""
                SELECT id, result_json
                FROM chat_message_records
                WHERE role = 'assistant'
                  AND result_json IS NOT NULL
                  AND result_json != ''
                  AND result_json != 'null'
                  AND id NOT IN (
                      SELECT record_id FROM parts_migration_log WHERE record_type = 'message'
                  )
                ORDER BY id
                LIMIT :batch OFFSET :offset
            """),
            {"batch": batch_size, "offset": offset},
        ).fetchall()
        if not rows:
            break
        for row in rows:
            yield row[0], row[1]
        offset += batch_size


def insert_parts(session: Session, message_id: str, parts: list[dict]) -> None:
    """将 parts 批量写入 chat_message_parts，并记录 migration_log。"""
    migration_key = str(uuid.uuid4())
    now = datetime.utcnow()

    for seq, part in enumerate(parts):
        part_id = str(uuid.uuid4())
        payload = {k: v for k, v in part.items() if k not in ("id", "server_id", "type", "state")}
        session.execute(
            text("""
                INSERT INTO chat_message_parts
                    (id, message_id, seq, type, state, payload_json, created_at, completed_at)
                VALUES
                    (:id, :message_id, :seq, :type, :state, :payload_json, :created_at, :completed_at)
            """),
            {
                "id": part_id,
                "message_id": message_id,
                "seq": seq,
                "type": part.get("type", "artifact"),
                "state": part.get("state", "done"),
                "payload_json": json.dumps(payload, ensure_ascii=False),
                "created_at": now,
                "completed_at": now,
            },
        )

    session.execute(
        text("""
            INSERT INTO parts_migration_log (record_id, record_type, migrated_at, parts_count)
            VALUES (:record_id, 'message', :migrated_at, :parts_count)
        """),
        {
            "record_id": message_id,
            "migrated_at": now,
            "parts_count": len(parts),
        },
    )


def count_pending(session: Session) -> int:
    row = session.execute(
        text("""
            SELECT COUNT(*)
            FROM chat_message_records
            WHERE role = 'assistant'
              AND result_json IS NOT NULL
              AND result_json != ''
              AND result_json != 'null'
              AND id NOT IN (
                  SELECT record_id FROM parts_migration_log WHERE record_type = 'message'
              )
        """)
    ).fetchone()
    return row[0] if row else 0


def ensure_tables(session: Session) -> None:
    """确保 parts_migration_log 表存在（幂等）。"""
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS parts_migration_log (
            id SERIAL PRIMARY KEY,
            record_id VARCHAR(64) NOT NULL,
            record_type VARCHAR(32) NOT NULL,
            migrated_at TIMESTAMP NOT NULL,
            parts_count INTEGER DEFAULT 0,
            UNIQUE(record_id, record_type)
        )
    """))
    session.commit()


def run(dry_run: bool = True, batch_size: int = 500, limit: Optional[int] = None) -> None:
    db_url = _get_connection_url()
    engine = create_engine(db_url, echo=False)
    SessionLocal = sessionmaker(bind=engine)

    session = SessionLocal()
    ensure_tables(session)

    pending = count_pending(session)
    if pending == 0:
        logger.info("没有需要迁移的消息，退出。")
        return

    logger.info(f"待迁移消息总数: {pending}")

    migrated = 0
    errors = 0
    sample_printed = 0

    for message_id, result_json in iter_messages_needing_migration(session, batch_size=batch_size):
        parts = _build_migrate_parts(result_json)
        if not parts:
            # 空结果也要记录，避免下次重复扫描
            if not dry_run:
                session.execute(
                    text("""
                        INSERT INTO parts_migration_log (record_id, record_type, migrated_at, parts_count)
                        VALUES (:record_id, 'message', :migrated_at, 0)
                        ON CONFLICT (record_id, record_type) DO NOTHING
                    """),
                    {"record_id": message_id, "migrated_at": datetime.utcnow()},
                )
            migrated += 1
            continue

        # Dry-run: 只打印前 3 条样例
        if dry_run and sample_printed < 3:
            logger.info(
                f"[DRY-RUN] message_id={message_id[:8]}... "
                f"parts_count={len(parts)} "
                f"types={[p['type'] for p in parts]}"
            )
            sample_printed += 1

        if not dry_run:
            try:
                insert_parts(session, message_id, parts)
                session.commit()
                migrated += 1
                if migrated % 500 == 0:
                    logger.info(f"已迁移 {migrated} 条")
            except Exception as exc:
                logger.error(f"迁移失败 message_id={message_id[:8]}: {exc}")
                session.rollback()
                errors += 1

        if limit and migrated >= limit:
            break

    if dry_run:
        logger.info(
            f"[DRY-RUN 完成] 本次会迁移 {pending} 条消息，"
            f"去掉 --dry-run 参数后执行实际写入。"
        )
    else:
        logger.info(f"迁移完成: 成功 {migrated} 条，失败 {errors} 条。")

    session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将旧 result_json 迁移到 chat_message_parts 表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="只打印待迁移数量和样例，不写入数据库（默认开启）",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="去掉 dry-run，真正写入数据库",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="每批查询条数（默认 500）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多迁移 N 条（用于测试）",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="DATABASE_URL（可选，默认读环境变量或本地 postgresql://...）",
    )
    args = parser.parse_args()

    if args.db_url:
        import os
        os.environ["DATABASE_URL"] = args.db_url

    try:
        run(dry_run=args.dry_run, batch_size=args.batch_size, limit=args.limit)
    except Exception as exc:
        logger.error(f"迁移脚本异常: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
