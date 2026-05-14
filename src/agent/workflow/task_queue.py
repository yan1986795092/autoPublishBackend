from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable, Optional
import asyncio
import uuid


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_INPUT = "awaiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PendingDecision:
    """等待用户决策的数据"""
    decision_id: str
    task_id: str
    question: str
    options: list[dict]
    context: dict = field(default_factory=dict)


@dataclass
class AsyncTask:
    """异步任务——对应一次完整的 Agent 编排请求"""
    id: str
    user_id: str
    prompt: str
    plan: Optional[dict] = None         # 预生成的执行计划（SSE 传递到 Worker）
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
    pending_decision: Optional[dict] = None

    def to_dict(self) -> dict:
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
        if self.pending_decision:
            d["pending_decision"] = {
                "decision_id": self.pending_decision.get("decision_id"),
                "question": self.pending_decision.get("question"),
                "options": self.pending_decision.get("options"),
                "context": self.pending_decision.get("context", {}),
            }
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
    event_type: str       # plan | thinking_step | progress | result | error | decision_required | llm_chunk
    data: dict

EventCallback = Callable[[StreamEvent], Awaitable[None]]

class EventBus:
    """轻量事件总线：支持多订阅者发布/订阅"""

    def __init__(self):
        self._subscribers: dict[str, list[EventCallback]] = {}  # task_id -> [callbacks]
        self._all_subscribers: list[EventCallback] = []          # 订阅所有任务的全局监听器

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
        # 发送给任务专属订阅者
        if event.task_id in self._subscribers:
            for cb in self._subscribers[event.task_id]:
                try:
                    await cb(event)
                except Exception:
                    pass
        # 发送给全局订阅者
        for cb in self._all_subscribers:
            try:
                await cb(event)
            except Exception:
                pass

# 全局单例
event_bus = EventBus()


class TaskQueue:
    """内存任务队列（单进程适用，可按需替换为 Redis）

    支持实时事件推送：通过 event_bus 通知所有订阅者（WebSocket / SSE）。
    """

    def __init__(self):
        self._queue: asyncio.Queue[AsyncTask] = asyncio.Queue()
        self._tasks: dict[str, AsyncTask] = {}
        self._progress_events: dict[str, list[asyncio.Event]] = {}
        self._decision_events: dict[str, asyncio.Event] = {}
        self._decisions: dict[str, Optional[str]] = {}
        # 死信队列：重试耗尽后仍然失败的任务
        self._dead_letter_queue: dict[str, AsyncTask] = {}

    async def enqueue(self, prompt: str, user_id: str, metadata: Optional[dict] = None, plan: Optional[dict] = None) -> AsyncTask:
        task = AsyncTask(
            id=str(uuid.uuid4()),
            user_id=user_id,
            prompt=prompt,
            plan=plan,
            metadata=metadata or {},
        )
        self._tasks[task.id] = task
        await self._queue.put(task)
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
            # 发布完成事件到 EventBus（SSE/WebSocket 订阅者会收到）
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
            asyncio.create_task(self._publish_event(StreamEvent(
                task_id=task_id,
                event_type="error",
                data={"message": error},
            )))

    def cancel_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task and task.state in (TaskState.PENDING, TaskState.RUNNING, TaskState.AWAITING_INPUT):
            task.state = TaskState.CANCELLED
            task.completed_at = datetime.utcnow().isoformat()
            self._notify_progress(task_id)
            return True
        return False

    def update_task_progress(self, task_id: str, progress: float, message: str = "", step: str = "") -> None:
        task = self._tasks.get(task_id)
        if task:
            task.update_progress(progress, message, step)
            task.completed_steps = int(progress / 100 * task.total_steps)
            self._notify_progress(task_id)
            # 发布进度事件（给 SSE 实时推送，无需轮询）
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
        """发布事件到 EventBus，异步执行不阻塞主流程"""
        try:
            await event_bus.publish(event)
        except Exception:
            pass

    def list_tasks(self, user_id: Optional[str] = None, limit: int = 20) -> list[AsyncTask]:
        self._cleanup_old_tasks()
        tasks = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
        if user_id:
            tasks = [t for t in tasks if t.user_id == user_id]
        return tasks[:limit]

    def send_to_dlq(self, task_id: str, reason: str = "") -> None:
        """将任务移入死信队列（所有重试耗尽后的最终归宿）"""
        task = self._tasks.get(task_id)
        if task:
            task.state = TaskState.FAILED
            task.error = f"DLQ: {reason or 'All retries exhausted'}"
            task.completed_at = datetime.utcnow().isoformat()
            self._dead_letter_queue[task_id] = task
            self._notify_progress(task_id)

    def get_dlq_tasks(self, limit: int = 20) -> list[AsyncTask]:
        """获取死信队列中的任务列表"""
        return sorted(self._dead_letter_queue.values(), key=lambda t: t.created_at, reverse=True)[:limit]

    def retry_from_dlq(self, task_id: str) -> Optional[AsyncTask]:
        """从死信队列中取出任务重新入队重试"""
        task = self._dead_letter_queue.pop(task_id, None)
        if task:
            task.state = TaskState.PENDING
            task.error = None
            task.completed_at = None
            task.retry_count = (task.retry_count or 0) + 1
            self._tasks[task_id] = task
            self._queue.put_nowait(task)
        return task

    async def publish_thinking_step(self, task_id: str, step_data: dict) -> None:
        """由 Worker 调用：在子任务完成时立即发布 thinking_step 事件"""
        await self._publish_event(StreamEvent(
            task_id=task_id,
            event_type="thinking_step",
            data=step_data,
        ))

    async def publish_llm_chunk(self, task_id: str, chunk_data: dict) -> None:
        """由 Worker 调用：发布 LLM 推理过程中的 token chunks（打字机效果）"""
        await self._publish_event(StreamEvent(
            task_id=task_id,
            event_type="llm_chunk",
            data=chunk_data,
        ))

    async def publish_reasoning(self, task_id: str, reasoning_data: dict) -> None:
        """由 Worker 调用：发布 LLM reasoning chain 片段"""
        await self._publish_event(StreamEvent(
            task_id=task_id,
            event_type="reasoning",
            data=reasoning_data,
        ))

    def _cleanup_old_tasks(self) -> None:
        """清理超过 1 小时的已完成/失败/取消任务，防止内存泄漏"""
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
            self._decision_events.pop(tid, None)
            self._decisions.pop(tid, None)

    # ========== 用户决策 ==========

    def set_decision_required(self, task_id: str, decision: PendingDecision) -> None:
        """标记任务需要用户输入，存储决策数据"""
        task = self._tasks.get(task_id)
        if not task:
            return
        task.state = TaskState.AWAITING_INPUT
        task.pending_decision = asdict(decision)
        self._decisions[task_id] = None
        self._notify_progress(task_id)
        # 发布决策请求事件
        asyncio.create_task(self._publish_event(StreamEvent(
            task_id=task_id,
            event_type="decision_required",
            data={
                "task_id": task_id,
                "decision_id": decision.decision_id,
                "question": decision.question,
                "options": decision.options,
                "context": decision.context,
            },
        )))

    def submit_decision(self, task_id: str, decision_id: str, choice: str) -> bool:
        """用户提交决策。返回 True 表示成功接受。"""
        task = self._tasks.get(task_id)
        if not task or task.state != TaskState.AWAITING_INPUT:
            return False
        if not task.pending_decision or task.pending_decision.get("decision_id") != decision_id:
            return False
        self._decisions[task_id] = choice
        task.pending_decision = None
        task.state = TaskState.RUNNING
        if task_id in self._decision_events:
            self._decision_events[task_id].set()
        self._notify_progress(task_id)
        return True

    async def wait_for_decision(self, task_id: str, timeout: float = 300.0) -> Optional[str]:
        """等待用户决策，返回选择值或 None（超时）"""
        if task_id not in self._decision_events:
            self._decision_events[task_id] = asyncio.Event()
        event = self._decision_events[task_id]
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self._decisions.get(task_id)
        except asyncio.TimeoutError:
            return None
        finally:
            self._decision_events.pop(task_id, None)
            self._decisions.pop(task_id, None)

    # 进度通知（用于 WebSocket）
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


# 全局单例
task_queue = TaskQueue()
