import asyncio
from types import SimpleNamespace

import pytest

from src.agent.workflow.task_queue import TaskState, task_queue
from src.agent.workflow.worker import AgentWorker


class _FakeRunResult:
    def __init__(self, status: str = "completed", payload: dict | None = None):
        self.status = SimpleNamespace(value=status)
        self._payload = payload or {"output": {"content": "ok"}}
        self.error = None

    def to_dict(self):
        return self._payload


@pytest.fixture(autouse=True)
def clear_global_queue():
    task_queue._tasks.clear()
    task_queue._dead_letter_queue.clear()
    while not task_queue._queue.empty():
        task_queue._queue.get_nowait()
    yield
    task_queue._tasks.clear()
    task_queue._dead_letter_queue.clear()
    while not task_queue._queue.empty():
        task_queue._queue.get_nowait()


@pytest.mark.asyncio
async def test_worker_does_not_run_cancelled_pending_task():
    called = {"run": 0}

    async def fake_run(**kwargs):
        called["run"] += 1
        return _FakeRunResult()

    worker = AgentWorker(orchestrator_factory=lambda **kw: SimpleNamespace(run=fake_run), max_retries=1)
    task = await task_queue.enqueue(prompt="test", user_id="u1")
    assert task_queue.cancel_task(task.id) is True

    dequeued = await task_queue.dequeue()
    assert dequeued is None
    assert called["run"] == 0
    assert task_queue.get_task(task.id).state == TaskState.CANCELLED


@pytest.mark.asyncio
async def test_worker_does_not_complete_task_cancelled_during_run():
    started = asyncio.Event()
    allow_finish = asyncio.Event()

    async def fake_run(**kwargs):
        started.set()
        await allow_finish.wait()
        return _FakeRunResult()

    worker = AgentWorker(orchestrator_factory=lambda **kw: SimpleNamespace(run=fake_run), max_retries=1)
    task = await task_queue.enqueue(prompt="test", user_id="u1")
    dequeued = await task_queue.dequeue()
    assert dequeued is not None

    runner = asyncio.create_task(worker._process_task(dequeued))
    await started.wait()
    assert task_queue.cancel_task(task.id) is True
    allow_finish.set()
    await runner

    current = task_queue.get_task(task.id)
    assert current.state == TaskState.CANCELLED
    assert current.result is None
