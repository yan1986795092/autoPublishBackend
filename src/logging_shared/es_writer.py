"""Elasticsearch 异步批量写入器（两服务共用）

将结构化 JSON 日志异步写入 Elasticsearch 的 logs-* 索引。
使用缓冲队列 + 定时批量写入，避免阻塞主请求链路。
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_es_client: Any | None = None
_queue: asyncio.Queue[dict] | None = None
_worker_task: asyncio.Task | None = None
_running = False


def _get_es_url() -> str:
    return os.getenv(
        "ELASTICSEARCH_URL",
        os.getenv("OTEL_ELASTICSEARCH_URL", "http://localhost:9200"),
    )


def _get_index_name() -> str:
    """按日期生成索引名：logs-YYYY.MM.DD"""
    return f"logs-{datetime.now(timezone.utc).strftime('%Y.%m.%d')}"


async def _get_es_client() -> Any:
    """延迟创建 ES 客户端。"""
    global _es_client
    if _es_client is None:
        from elasticsearch import AsyncElasticsearch
        url = _get_es_url()
        try:
            _es_client = AsyncElasticsearch(
                hosts=[url],
                request_timeout=30,
                retry_on_timeout=True,
                max_retries=3,
            )
            info = await _es_client.info()
            logger.info("ES 写入器已连接到 Elasticsearch %s", info["version"]["number"])
        except Exception as e:
            logger.warning("无法连接到 Elasticsearch: %s", e)
            _es_client = None
    return _es_client


async def _worker(batch_size: int = 100, flush_interval: float = 5.0) -> None:
    """后台写入协程：从队列消费日志，批量写入 ES。"""
    global _running
    _running = True
    buffer: list[dict] = []
    last_flush = asyncio.get_event_loop().time()

    while _running or buffer:
        try:
            item = await asyncio.wait_for(
                _queue.get(),
                timeout=flush_interval,
            ) if _running else await _queue.get()
            buffer.append(item)
            _queue.task_done()
        except asyncio.TimeoutError:
            pass

        now = asyncio.get_event_loop().time()
        if buffer and (len(buffer) >= batch_size or (now - last_flush) >= flush_interval):
            await _flush_buffer(buffer)
            buffer.clear()
            last_flush = now

    if buffer:
        await _flush_buffer(buffer)


async def _flush_buffer(buffer: list[dict]) -> None:
    """将一批日志写入 ES。"""
    if not buffer:
        return
    es = await _get_es_client()
    if not es:
        return

    index = _get_index_name()
    operations = []
    for doc in buffer:
        doc = dict(doc)
        if "@timestamp" not in doc:
            doc["@timestamp"] = datetime.now(timezone.utc).isoformat()
        operations.append({"index": {"_index": index}})
        operations.append(doc)

    try:
        await es.bulk(operations=operations, refresh=False)
    except Exception as e:
        logger.error("ES 批量写入失败: %s", e)


async def start_writer() -> None:
    """启动后台写入器（应用启动时调用一次）。"""
    global _queue, _worker_task
    if _worker_task is not None:
        return
    _queue = asyncio.Queue(maxsize=10000)
    _worker_task = asyncio.create_task(_worker())
    logger.info("日志 ES 写入器已启动")


async def stop_writer() -> None:
    """停止后台写入器（应用关闭时调用）。"""
    global _running, _worker_task, _es_client
    _running = False
    if _worker_task:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        _worker_task = None
    if _es_client:
        await _es_client.close()
        _es_client = None


def emit(record: dict) -> None:
    """将一条日志记录加入写入队列（异步，非阻塞）。"""
    if _queue is None:
        return
    try:
        _queue.put_nowait(record)
    except asyncio.QueueFull:
        pass
