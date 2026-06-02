"""快照持久化存储：JSON 文件系统 + SQLite 元数据索引"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import ExecutionSnapshot

logger = logging.getLogger(__name__)


class SnapshotStorage:
    """快照存储层

    索引字段写入 SQLite 便于检索，完整 JSON 快照存入文件系统。
    """

    def __init__(self, storage_dir: str = "snapshots", db_session=None):
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._db = db_session

    def save(self, snapshot: ExecutionSnapshot) -> str:
        """保存快照到文件系统。"""
        month_dir = self._storage_dir / snapshot.timestamp.strftime("%Y%m")
        month_dir.mkdir(parents=True, exist_ok=True)

        json_path = month_dir / f"{snapshot.snapshot_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, ensure_ascii=False, indent=2, default=str)

        return str(json_path)

    def load(self, snapshot_id: str) -> Optional[ExecutionSnapshot]:
        """按 snapshot_id 加载快照。"""
        for month_dir in sorted(self._storage_dir.iterdir(), reverse=True):
            if not month_dir.is_dir():
                continue
            json_path = month_dir / f"{snapshot_id}.json"
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ExecutionSnapshot.from_dict(data)
        return None

    def list(self, *, agent_name: str = None,
             template_name: str = None,
             template_version: str = None,
             start_time: datetime = None,
             end_time: datetime = None,
             tags: list[str] = None,
             limit: int = 50) -> list[ExecutionSnapshot]:
        """按条件检索快照。"""
        summaries = []
        for month_dir in sorted(self._storage_dir.iterdir(), reverse=True):
            if not month_dir.is_dir():
                continue
            for fpath in sorted(month_dir.iterdir(), reverse=True):
                if not fpath.name.endswith(".json"):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue

                if agent_name and data.get("agent_name") != agent_name:
                    continue
                if tags and not all(t in data.get("tags", []) for t in tags):
                    continue
                if start_time:
                    ts = data.get("timestamp", "")
                    if isinstance(ts, str) and ts < start_time.isoformat():
                        continue
                if end_time:
                    ts = data.get("timestamp", "")
                    if isinstance(ts, str) and ts > end_time.isoformat():
                        continue

                summaries.append(data)
                if len(summaries) >= limit:
                    break
            if len(summaries) >= limit:
                break

        return [ExecutionSnapshot.from_dict(s) for s in summaries]
