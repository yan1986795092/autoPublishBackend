"""TaskQueue 单元测试：入队、出队、进度、通知"""
import pytest
import asyncio
from src.agent.workflow.task_queue import TaskQueue, TaskState


@pytest.fixture
def queue():
    return TaskQueue()


class TestTaskQueueBasic:
    @pytest.mark.asyncio
    async def test_enqueue_creates_task(self, queue):
        task = await queue.enqueue(prompt="test prompt", user_id="user-1")
        assert task.id is not None
        assert task.prompt == "test prompt"
        assert task.user_id == "user-1"
        assert task.state == TaskState.PENDING

    @pytest.mark.asyncio
    async def test_enqueue_sets_metadata(self, queue):
        task = await queue.enqueue(
            prompt="test", user_id="user-1",
            metadata={"source": "api"}
        )
        assert task.metadata == {"source": "api"}

    @pytest.mark.asyncio
    async def test_dequeue_returns_task(self, queue):
        await queue.enqueue(prompt="task-1", user_id="user-1")
        task = await queue.dequeue()
        assert task is not None
        assert task.prompt == "task-1"
        assert task.state == TaskState.RUNNING

    @pytest.mark.asyncio
    async def test_dequeue_empty_returns_none(self, queue):
        task = await queue.dequeue()
        assert task is None

    @pytest.mark.asyncio
    async def test_dequeue_respects_fifo(self, queue):
        await queue.enqueue(prompt="first", user_id="u1")
        await queue.enqueue(prompt="second", user_id="u1")
        t1 = await queue.dequeue()
        t2 = await queue.dequeue()
        assert t1.prompt == "first"
        assert t2.prompt == "second"


class TestTaskQueueState:
    @pytest.mark.asyncio
    async def test_complete_task(self, queue):
        t = await queue.enqueue(prompt="test", user_id="u1")
        await queue.dequeue()
        queue.complete_task(t.id, {"result": "ok"})
        task = queue.get_task(t.id)
        assert task.state == TaskState.COMPLETED
        assert task.progress == 100.0
        assert task.result == {"result": "ok"}
        assert task.completed_at is not None

    @pytest.mark.asyncio
    async def test_fail_task(self, queue):
        t = await queue.enqueue(prompt="test", user_id="u1")
        queue.fail_task(t.id, "error occurred")
        task = queue.get_task(t.id)
        assert task.state == TaskState.FAILED
        assert task.error == "error occurred"

    @pytest.mark.asyncio
    async def test_update_progress(self, queue):
        t = await queue.enqueue(prompt="test", user_id="u1")
        queue.update_task_progress(t.id, 50.0, message="halfway", step="processing")
        task = queue.get_task(t.id)
        assert task.progress == 50.0
        assert task.progress_message == "halfway"
        assert task.current_step == "processing"

    @pytest.mark.asyncio
    async def test_get_nonexistent_task(self, queue):
        task = queue.get_task("nonexistent")
        assert task is None

    @pytest.mark.asyncio
    async def test_dequeue_skips_cancelled_pending_task(self, queue):
        cancelled = await queue.enqueue(prompt="cancelled", user_id="u1")
        active = await queue.enqueue(prompt="active", user_id="u1")
        assert queue.cancel_task(cancelled.id) is True

        task = await queue.dequeue()
        assert task is not None
        assert task.id == active.id
        assert queue.get_task(cancelled.id).state == TaskState.CANCELLED

    @pytest.mark.asyncio
    async def test_retry_from_dlq_clears_stale_runtime_state(self, queue):
        task = await queue.enqueue(prompt="test", user_id="u1")
        task.state = TaskState.FAILED
        task.progress = 72.0
        task.progress_message = "old progress"
        task.current_step = "old-step"
        task.completed_steps = 3
        task.result = {"content": "stale"}
        queue._dead_letter_queue[task.id] = task

        retried = await queue.retry_from_dlq(task.id)

        assert retried is not None
        assert retried.state == TaskState.PENDING
        assert retried.progress == 0.0
        assert retried.progress_message == ""
        assert retried.current_step == ""
        assert retried.completed_steps == 0
        assert retried.result is None
        assert retried.error is None


class TestTaskQueueListing:
    @pytest.mark.asyncio
    async def test_list_tasks_empty(self, queue):
        tasks = queue.list_tasks()
        assert tasks == []

    @pytest.mark.asyncio
    async def test_list_tasks_ordered_by_date(self, queue):
        await queue.enqueue(prompt="older", user_id="u1")
        await asyncio.sleep(0.01)
        await queue.enqueue(prompt="newer", user_id="u1")
        tasks = queue.list_tasks()
        assert tasks[0].prompt == "newer"

    @pytest.mark.asyncio
    async def test_list_tasks_filter_by_user(self, queue):
        await queue.enqueue(prompt="alice-task", user_id="alice")
        await queue.enqueue(prompt="bob-task", user_id="bob")
        alice_tasks = queue.list_tasks(user_id="alice")
        assert len(alice_tasks) == 1
        assert alice_tasks[0].user_id == "alice"

    @pytest.mark.asyncio
    async def test_list_tasks_limit(self, queue):
        for i in range(5):
            await queue.enqueue(prompt=f"task-{i}", user_id="u1")
        tasks = queue.list_tasks(limit=3)
        assert len(tasks) == 3


class TestTaskQueueProgressEvents:
    @pytest.mark.asyncio
    async def test_wait_for_progress(self, queue):
        t = await queue.enqueue(prompt="test", user_id="u1")

        async def delayed_update():
            await asyncio.sleep(0.05)
            queue.update_task_progress(t.id, 50.0, message="update")

        asyncio.create_task(delayed_update())
        result = await queue.wait_for_progress(t.id, timeout=1.0)
        assert result is not None
        assert result["progress"] == 50.0

    @pytest.mark.asyncio
    async def test_wait_for_progress_timeout(self, queue):
        t = await queue.enqueue(prompt="test", user_id="u1")
        result = await queue.wait_for_progress(t.id, timeout=0.1)
        assert result is None
