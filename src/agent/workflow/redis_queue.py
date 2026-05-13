"""Redis 任务队列 — 生产环境多进程/多机器部署支持

替代默认的 in-memory TaskQueue，使用 Redis 作为消息代理。
启用方式：设置环境变量 REDIS_URL。
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisTaskQueue:
    """Redis 支持的任务队列

    结构：
      - agent:task:{task_id} → Hash (任务元数据)
      - agent:queue → List (待处理队列)
      - agent:task_list → SortedSet (按时间排序的任务列表)
      - agent:decision:{task_id} → Hash (用户决策数据)
    """

    TASK_PREFIX = "agent:task"
    QUEUE_KEY = "agent:queue"
    TASK_LIST_KEY = "agent:task_list"
    DECISION_PREFIX = "agent:decision"
    TASK_TTL = 86400 * 90  # 90 天

    def __init__(self, redis_url: Optional[str] = None):
        self._redis_url = redis_url or os.getenv("REDIS_URL", "")
        self._redis: Optional[aioredis.Redis] = None
        self._available = bool(self._redis_url) and REDIS_AVAILABLE

    async def _get_redis(self) -> Optional[aioredis.Redis]:
        if not self._available:
            return None
        if self._redis is None:
            try:
                self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
                await self._redis.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, falling back to in-memory queue")
                self._available = False
                return None
        return self._redis

    async def enqueue(self, prompt: str, user_id: str) -> dict:
        """添加任务到队列"""
        r = await self._get_redis()
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        task = {
            "id": task_id,
            "user_id": user_id,
            "prompt": prompt,
            "state": "pending",
            "progress": 0,
            "progress_message": "",
            "current_step": "",
            "total_steps": 0,
            "completed_steps": 0,
            "created_at": now,
        }

        if r:
            pipe = r.pipeline()
            pipe.hset(f"{self.TASK_PREFIX}:{task_id}", mapping=task)
            pipe.lpush(self.QUEUE_KEY, task_id)
            pipe.zadd(self.TASK_LIST_KEY, {task_id: datetime.now(timezone.utc).timestamp()})
            pipe.expire(f"{self.TASK_PREFIX}:{task_id}", self.TASK_TTL)
            await pipe.execute()
            return task

        # Fallback to in-memory
        from .task_queue import task_queue
        async_task = await task_queue.enqueue(prompt=prompt, user_id=user_id)
        return async_task.to_dict()

    async def dequeue(self) -> Optional[dict]:
        """从队列取出下一个任务"""
        r = await self._get_redis()
        if r:
            task_id = await r.rpop(self.QUEUE_KEY)
            if task_id:
                task_data = await r.hgetall(f"{self.TASK_PREFIX}:{task_id}")
                if task_data:
                    task_data["state"] = "running"
                    await r.hset(f"{self.TASK_PREFIX}:{task_id}", mapping=task_data)
                    return task_data
            return None

        from .task_queue import task_queue
        try:
            async_task = await asyncio.wait_for(task_queue.dequeue(), timeout=1.0)
            return async_task.to_dict() if async_task else None
        except Exception:
            return None

    async def get_task(self, task_id: str) -> Optional[dict]:
        """获取任务详情"""
        r = await self._get_redis()
        if r:
            data = await r.hgetall(f"{self.TASK_PREFIX}:{task_id}")
            return data if data else None

        from .task_queue import task_queue
        task = task_queue.get_task(task_id)
        return task.to_dict() if task else None

    async def update_progress(self, task_id: str, progress: float, message: str = "", step: str = ""):
        """更新任务进度"""
        r = await self._get_redis()
        if r:
            updates = {"progress": str(progress), "progress_message": message, "current_step": step}
            await r.hset(f"{self.TASK_PREFIX}:{task_id}", mapping=updates)
            return

        from .task_queue import task_queue
        task_queue.update_task_progress(task_id, progress, message=message, step=step)

    async def complete_task(self, task_id: str, result: dict):
        """标记任务完成"""
        r = await self._get_redis()
        if r:
            now = datetime.now(timezone.utc).isoformat()
            await r.hset(f"{self.TASK_PREFIX}:{task_id}", mapping={
                "state": "completed",
                "progress": "100",
                "completed_at": now,
                "result": json.dumps(result, ensure_ascii=False),
            })
            return

        from .task_queue import task_queue
        task_queue.complete_task(task_id, result)

    async def list_tasks(self, user_id: str = "", limit: int = 20) -> list[dict]:
        """列出任务"""
        r = await self._get_redis()
        if r:
            task_ids = await r.zrevrange(self.TASK_LIST_KEY, 0, limit - 1)
            tasks = []
            for tid in task_ids:
                data = await r.hgetall(f"{self.TASK_PREFIX}:{tid}")
                if data:
                    if not user_id or data.get("user_id") == user_id:
                        tasks.append(data)
            return tasks

        from .task_queue import task_queue
        return [t.to_dict() for t in task_queue.list_tasks(user_id=user_id, limit=limit)]

    @property
    def available(self) -> bool:
        return self._available


# 模块级单例
redis_task_queue = RedisTaskQueue()
