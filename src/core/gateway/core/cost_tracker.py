import asyncio
import aiosqlite
from datetime import datetime
from typing import Optional
from dataclasses import dataclass


@dataclass
class CostRecord:
    timestamp: datetime
    trace_id: str
    source_app: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    estimated_cost: float


class CostTracker:
    def __init__(
        self,
        db_path: str,
        batch_size: int = 50,
        flush_interval: float = 5.0,
        pricing: Optional[dict] = None,
    ):
        self.db_path = db_path
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.pricing = pricing or {}
        self._queue: asyncio.Queue[CostRecord] = asyncio.Queue()
        self._started = False
        self._flush_task: Optional[asyncio.Task] = None

    async def start(self):
        if self._started:
            return
        self._started = True
        await self._init_db()
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self):
        if not self._started:
            return
        await self._flush_batch()
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        self._started = False

    async def _init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cost_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    source_app TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    estimated_cost REAL NOT NULL
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_cost_timestamp ON cost_records(timestamp)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_cost_provider ON cost_records(provider)
            """)
            await db.commit()

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        prices = self.pricing.get(model, {"input": 0.0, "output": 0.0})
        return (prompt_tokens / 1000 * prices["input"]) + (completion_tokens / 1000 * prices["output"])

    async def record(self, record: CostRecord):
        await self._queue.put(record)

    async def _flush_loop(self):
        while True:
            await asyncio.sleep(self.flush_interval)
            await self._flush_batch()

    async def _flush_batch(self):
        records = []
        while not self._queue.empty() and len(records) < self.batch_size:
            try:
                records.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if not records:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany("""
                INSERT INTO cost_records
                (timestamp, trace_id, source_app, provider, model, prompt_tokens, completion_tokens, latency_ms, estimated_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (r.timestamp.isoformat(), r.trace_id, r.source_app, r.provider, r.model,
                 r.prompt_tokens, r.completion_tokens, r.latency_ms, r.estimated_cost)
                for r in records
            ])
            await db.commit()

    async def query_stats(
        self,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        provider: Optional[str] = None,
    ) -> dict:
        conditions = []
        params = []
        if from_time:
            conditions.append("timestamp >= ?")
            params.append(from_time.isoformat())
        if to_time:
            conditions.append("timestamp <= ?")
            params.append(to_time.isoformat())
        if provider:
            conditions.append("provider = ?")
            params.append(provider)
        where = " AND ".join(conditions) if conditions else "1=1"

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(f"""
                SELECT
                    COUNT(*) as total_requests,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(estimated_cost) as total_cost,
                    AVG(latency_ms) as avg_latency_ms
                FROM cost_records
                WHERE {where}
            """, params)
            row = await cursor.fetchone()
            return {
                "total_requests": row["total_requests"] or 0,
                "total_prompt_tokens": row["total_prompt_tokens"] or 0,
                "total_completion_tokens": row["total_completion_tokens"] or 0,
                "total_cost": round(row["total_cost"] or 0.0, 6),
                "avg_latency_ms": round(row["avg_latency_ms"] or 0.0, 2),
            }
