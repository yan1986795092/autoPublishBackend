"""统一的异步 ES 写入器：管理 traces/events/logs 队列 + JSONL 回退。

用法：
    writer = UnifiedEsWriter.get_instance()
    await writer.start()
    await writer.write("traces", span_dict)
    await writer.write("events", event_dict)
    await writer.write("logs", log_dict)
    await writer.stop()
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_logger

logger = get_logger("logging.unified_writer")

_INDEX_PREFIXES = {
    "traces": "traces",
    "events": "agent-events",
    "logs": "logs",
}


class UnifiedEsWriter:
    """单实例异步 ES 写入器，支持按索引队列和文件回退。"""

    _instance: UnifiedEsWriter | None = None

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {
            "traces": asyncio.Queue(maxsize=10000),
            "events": asyncio.Queue(maxsize=5000),
            "logs": asyncio.Queue(maxsize=10000),
        }
        self._running = False
        self._worker_task: asyncio.Task | None = None
        self._es_available = False
        self._es_client: Any | None = None
        self._fallback_dir = Path("logs")
        self._fallback_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls) -> UnifiedEsWriter:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def write(self, index_type: str, doc: dict) -> None:
        """将文档加入队列等待异步 ES 写入。

        Args:
            index_type: "traces" | "events" | "logs"
            doc: 文档字典（必须可 JSON 序列化）
        """
        queue = self._queues.get(index_type)
        if queue is None:
            logger.warning("未知的索引类型: %s", index_type)
            return
        try:
            queue.put_nowait(dict(doc))
        except asyncio.QueueFull:
            # 队列满时不丢弃，直接写 fallback 文件
            logger.warning("队列 %s 已满，写入回退文件", index_type)
            self._write_fallback(index_type, [dict(doc)])

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("UnifiedEsWriter 已启动（队列: traces/events/logs）")

    async def stop(self) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        if self._es_client:
            try:
                await self._es_client.close()
            except Exception:
                pass
            self._es_client = None
        logger.info("UnifiedEsWriter 已停止")

    async def _worker_loop(self) -> None:
        """主循环：轮询所有队列，批量写满或超时时刷新。"""
        buffer: dict[str, list[dict]] = {k: [] for k in self._queues}
        BATCH_SIZE = 100
        FLUSH_INTERVAL = 5.0
        last_flush = time.monotonic()

        while self._running or any(len(buf) > 0 for buf in buffer.values()):
            # 轮询各队列
            for qtype, queue in self._queues.items():
                for _ in range(BATCH_SIZE):
                    try:
                        item = queue.get_nowait()
                        buffer[qtype].append(item)
                    except asyncio.QueueEmpty:
                        break

            # 检查刷新条件
            now = time.monotonic()
            timed_out = (now - last_flush) >= FLUSH_INTERVAL
            for qtype, buf in buffer.items():
                if buf and (len(buf) >= BATCH_SIZE or timed_out):
                    await self._flush(qtype, buf)
                    buffer[qtype] = []

            if timed_out:
                last_flush = now

            await asyncio.sleep(0.1)

        # 关闭前最终刷新
        for qtype, buf in buffer.items():
            if buf:
                await self._flush(qtype, buf)

    async def _flush(self, index_type: str, docs: list[dict]) -> None:
        """尝试 ES 写入；失败时回退到 JSONL 文件。"""
        if await self._try_es_bulk(index_type, docs):
            return
        self._write_fallback(index_type, docs)

    async def _try_es_bulk(self, index_type: str, docs: list[dict]) -> bool:
        """批量写入 ES。成功返回 True。"""
        if not self._es_available:
            return False
        es = await self._get_es_client()
        if not es:
            self._es_available = False
            return False

        index_name = self._index_name(index_type)
        operations: list[dict] = []
        for doc in docs:
            if "@timestamp" not in doc:
                doc["@timestamp"] = datetime.now(timezone.utc).isoformat()
            operations.append({"index": {"_index": index_name}})
            operations.append(doc)

        try:
            resp = await es.bulk(operations=operations, refresh=False)
            if resp.get("errors"):
                logger.warning("ES 批量写入 %s 存在错误", index_type)
            return True
        except Exception as exc:
            logger.warning("ES 批量写入 %s 失败: %s", index_type, exc)
            self._es_available = False
            return False

    def _write_fallback(self, index_type: str, docs: list[dict]) -> None:
        """将文档追加到 JSONL 回退文件。"""
        path = self._fallback_dir / f"{index_type}-fallback.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                for doc in docs:
                    f.write(json.dumps(doc, default=str, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("无法写入回退文件 %s: %s", path, exc)

    def _index_name(self, index_type: str) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y.%m.%d")
        prefix = _INDEX_PREFIXES.get(index_type, index_type)
        return f"{prefix}-{date_str}"

    async def _get_es_client(self) -> Any | None:
        """首次使用时延迟初始化 ES 客户端。"""
        if self._es_client is not None:
            return self._es_client
        from elasticsearch import AsyncElasticsearch

        url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        try:
            client = AsyncElasticsearch(
                hosts=[url],
                request_timeout=30,
                retry_on_timeout=True,
                max_retries=3,
            )
            info = await client.info()
            self._es_client = client
            self._es_available = True
            logger.info("UnifiedEsWriter 已连接到 ES %s", info["version"]["number"])
            return self._es_client
        except Exception as exc:
            logger.warning("UnifiedEsWriter 无法连接到 ES %s: %s", url, exc)
            self._es_client = None
            self._es_available = False
            return None
