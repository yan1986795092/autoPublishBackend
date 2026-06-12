from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable, Optional
import asyncio
import structlog
import uuid

logger = structlog.get_logger(__name__)


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AsyncTask:
    """异步任务——对应一次完整的 Agent 编排请求"""
    id: str
    user_id: str
    prompt: str
    plan: Optional[dict] = None
    skill_context: Optional[dict] = None
    state: TaskState = TaskState.PENDING
    progress: float = 0.0
    progress_message: str = ""
    current_step: str = ""
    total_steps: int = 1
    completed_steps: int = 0
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    retry_count: int = 0

    def to_dict(self) -> dict:
        """转换为字典格式"""
        d: dict[str, Any] = {
            "id": self.id,
            "user_id": self.user_id,
            "prompt": self.prompt[:100],
            "state": self.state.value,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }
        if self.plan:
            d["plan"] = {k: v for k, v in self.plan.items() if k in ("goal", "reasoning", "estimated_steps")}
        return d

    def update_progress(self, progress: float, message: str = "", step: str = "") -> None:
        self.progress = min(progress, 100.0)
        if message:
            self.progress_message = message
        if step:
            self.current_step = step


# ──────────────────────────────────────────
# 全局实时事件总线（供 SSE + WebSocket 订阅）
# ──────────────────────────────────────────

@dataclass
class StreamEvent:
    """流式事件结构"""
    task_id: str
    event_type: str       # 事件类型: plan | thinking_step | progress | result | error | llm_chunk
    data: dict

EventCallback = Callable[[StreamEvent], Awaitable[None]]

class EventBus:
    """轻量事件总线：支持多订阅者发布/订阅"""

    def __init__(self):
        self._subscribers: dict[str, list[EventCallback]] = {}
        self._all_subscribers: list[EventCallback] = []

    def subscribe(self, task_id: str, callback: EventCallback) -> None:
        if task_id not in self._subscribers:
            self._subscribers[task_id] = []
        self._subscribers[task_id].append(callback)

    def subscribe_all(self, callback: EventCallback) -> None:
        self._all_subscribers.append(callback)

    def unsubscribe(self, task_id: str, callback: EventCallback) -> None:
        if task_id in self._subscribers:
            try:
                self._subscribers[task_id].remove(callback)
            except ValueError:
                pass

    async def publish(self, event: StreamEvent) -> None:
        if event.task_id in self._subscribers:
            for cb in self._subscribers[event.task_id]:
                try:
                    await cb(event)
                except Exception as e:
                    logger.warning(
                        "EventBus 订阅者执行失败",
                        _type="system",
                        event_name="subscriber_error",
                        message=f"EventBus 订阅者错误，任务 {event.task_id[:8]}: {e}",
                        status="error",
                    )
        for cb in self._all_subscribers:
            try:
                await cb(event)
            except Exception as e:
                logger.warning(
                    "EventBus 全局订阅者执行失败",
                    _type="system",
                    event_name="subscriber_error",
                    message=f"EventBus 全局订阅者错误: {e}",
                    status="error",
                )

event_bus = EventBus()


class TaskQueue:
    """内存任务队列（单进程适用，可按需替换为 Redis）"""

    def __init__(self):
        self._queue: asyncio.Queue[AsyncTask] = asyncio.Queue()
        self._tasks: dict[str, AsyncTask] = {}
        self._progress_events: dict[str, list[asyncio.Event]] = {}
        self._dead_letter_queue: dict[str, AsyncTask] = {}

    async def enqueue(self, prompt: str, user_id: str, metadata: Optional[dict] = None, plan: Optional[dict] = None, skill_context: Optional[dict] = None) -> AsyncTask:
        task = AsyncTask(
            id=str(uuid.uuid4()),
            user_id=user_id,
            prompt=prompt,
            plan=plan,
            skill_context=skill_context,
            metadata=metadata or {},
        )
        self._tasks[task.id] = task
        await self._queue.put(task)
        logger.info(
            "任务已入队",
            _type="system",
            event_name="task_enqueued",
            message=f"任务 {task.id[:8]} 已入队，用户 {user_id[:8]}",
            status="pending",
        )
        return task

    async def dequeue(self) -> Optional[AsyncTask]:
        try:
            task = await asyncio.wait_for(self._queue.get(), timeout=30.0)
            task.state = TaskState.RUNNING
            return task
        except asyncio.TimeoutError:
            return None

    def get_task(self, task_id: str) -> Optional[AsyncTask]:
        return self._tasks.get(task_id)

    def complete_task(self, task_id: str, result: dict) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.state = TaskState.COMPLETED
            task.progress = 100.0
            task.result = result
            task.completed_at = datetime.utcnow().isoformat()
            self._notify_progress(task_id)
            logger.info(
                "任务已完成",
                _type="system",
                event_name="task_completed",
                message=f"任务 {task_id[:8]} 已完成",
                status="ok",
            )
            asyncio.create_task(self._publish_event(StreamEvent(
                task_id=task_id,
                event_type="result",
                data=result,
            )))

    def fail_task(self, task_id: str, error: str) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.state = TaskState.FAILED
            task.error = error
            task.completed_at = datetime.utcnow().isoformat()
            self._notify_progress(task_id)
            logger.warning(
                "任务失败",
                _type="system",
                event_name="task_failed",
                message=f"任务 {task_id[:8]} 失败: {error[:80]}",
                status="failed",
            )
            asyncio.create_task(self._publish_event(StreamEvent(
                task_id=task_id,
                event_type="error",
                data={"message": error},
            )))

    def cancel_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task and task.state in (TaskState.PENDING, TaskState.RUNNING):
            task.state = TaskState.CANCELLED
            task.completed_at = datetime.utcnow().isoformat()
            self._notify_progress(task_id)
            logger.info(
                "任务已取消",
                _type="system",
                event_name="task_cancelled",
                message=f"任务 {task_id[:8]} 已取消",
                status="stopped",
            )
            return True
        return False

    def update_task_progress(self, task_id: str, progress: float, message: str = "", step: str = "") -> None:
        task = self._tasks.get(task_id)
        if task:
            task.update_progress(progress, message, step)
            task.completed_steps = int(progress / 100 * task.total_steps)
            self._notify_progress(task_id)
            asyncio.create_task(self._publish_event(StreamEvent(
                task_id=task_id,
                event_type="progress",
                data={
                    "progress": task.progress,
                    "current_step": task.current_step,
                    "progress_message": task.progress_message,
                    "completed_steps": task.completed_steps,
                    "total_steps": task.total_steps,
                },
            )))

    async def _publish_event(self, event: StreamEvent) -> None:
        try:
            await event_bus.publish(event)
        except Exception as e:
            logger.warning(
                "事件发布失败",
                _type="system",
                event_name="publish_failed",
                message=f"事件发布失败: {e}",
                status="error",
            )

    def list_tasks(self, user_id: Optional[str] = None, limit: int = 20) -> list[AsyncTask]:
        self._cleanup_old_tasks()
        tasks = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
        if user_id:
            tasks = [t for t in tasks if t.user_id == user_id]
        return tasks[:limit]

    def send_to_dlq(self, task_id: str, reason: str = "") -> None:
        task = self._tasks.get(task_id)
        if task:
            task.state = TaskState.FAILED
            task.error = f"DLQ: {reason or '所有重试已耗尽'}"
            task.completed_at = datetime.utcnow().isoformat()
            self._dead_letter_queue[task_id] = task
            self._notify_progress(task_id)
            logger.warning(
                "任务进入死信队列",
                _type="system",
                event_name="dlq",
                message=f"任务 {task_id[:8]} 已发送至死信队列: {reason}",
                status="failed",
            )

    def get_dlq_tasks(self, limit: int = 20) -> list[AsyncTask]:
        return sorted(self._dead_letter_queue.values(), key=lambda t: t.created_at, reverse=True)[:limit]

    async def retry_from_dlq(self, task_id: str) -> Optional[AsyncTask]:
        task = self._dead_letter_queue.pop(task_id, None)
        if task:
            task.state = TaskState.PENDING
            task.error = None
            task.completed_at = None
            task.retry_count = (task.retry_count or 0) + 1
            self._tasks[task_id] = task
            await self._queue.put(task)
        return task

    async def publish_thinking_step(self, task_id: str, step_data: dict) -> None:
        await self._publish_event(StreamEvent(
            task_id=task_id,
            event_type="thinking_step",
            data=step_data,
        ))

    async def publish_llm_chunk(self, task_id: str, chunk_data: dict) -> None:
        await self._publish_event(StreamEvent(
            task_id=task_id,
            event_type="llm_chunk",
            data=chunk_data,
        ))

    async def publish_reasoning(self, task_id: str, reasoning_data: dict) -> None:
        await self._publish_event(StreamEvent(
            task_id=task_id,
            event_type="reasoning",
            data=reasoning_data,
        ))

    def _cleanup_old_tasks(self) -> None:
        cutoff = datetime.utcnow().timestamp() - 3600
        stale = [
            tid for tid, t in self._tasks.items()
            if t.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED)
            and t.completed_at
            and datetime.fromisoformat(t.completed_at).timestamp() < cutoff
        ]
        for tid in stale:
            self._tasks.pop(tid, None)
            self._progress_events.pop(tid, None)

    def _notify_progress(self, task_id: str) -> None:
        if task_id in self._progress_events:
            for event in self._progress_events[task_id]:
                event.set()

    async def wait_for_progress(self, task_id: str, timeout: float = 30.0) -> Optional[dict]:
        if task_id not in self._progress_events:
            self._progress_events[task_id] = []
        event = asyncio.Event()
        self._progress_events[task_id].append(event)
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            task = self.get_task(task_id)
            return task.to_dict() if task else None
        except asyncio.TimeoutError:
            return None
        finally:
            self._progress_events[task_id].remove(event)
            if not self._progress_events[task_id]:
                del self._progress_events[task_id]


task_queue = TaskQueue()
