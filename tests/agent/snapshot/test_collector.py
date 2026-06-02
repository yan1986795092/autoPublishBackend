import pytest
import time
from datetime import datetime
from src.agent.snapshot.collector import SnapshotCollector, set_collector, get_collector
from src.agent.snapshot.models import PromptMessageSnapshot


class TestSnapshotCollector:
    def test_create_and_finalize(self):
        collector = SnapshotCollector(
            agent_name="WritingAgent",
            profile={"name": "WritingAgent", "role": "内容创作者"},
            task_id="task_001",
        )
        time.sleep(0.01)
        snapshot = collector.finalize({"status": "completed"})
        assert snapshot.agent_name == "WritingAgent"
        assert snapshot.task_id == "task_001"
        assert snapshot.duration_ms > 0
        assert snapshot.result == {"status": "completed"}

    def test_record_prompt_message(self):
        collector = SnapshotCollector("WritingAgent", {"name": "test"})
        collector.record_prompt_message(PromptMessageSnapshot(
            role="system",
            template_name="system.base",
            template_version="v2.0",
            template_hash="abc",
            source="template",
            resolved_content="You are a writer...",
        ))
        collector.record_prompt_message(PromptMessageSnapshot(
            role="user",
            source="memory",
            resolved_content="some memories",
        ))
        snapshot = collector.finalize({})
        assert len(snapshot.prompt_messages) == 2
        assert snapshot.prompt_messages[0].template_name == "system.base"

    def test_record_tool_call(self):
        collector = SnapshotCollector("WritingAgent", {"name": "test"})
        collector.record_tool_call("web_search", "query: AI trends", "3 results found")
        snapshot = collector.finalize({})
        assert len(snapshot.context.tool_calls) == 1
        assert snapshot.context.tool_calls[0]["tool"] == "web_search"

    def test_record_llm_call(self):
        collector = SnapshotCollector("WritingAgent", {"name": "test"})
        collector.record_llm_call(
            model="gpt-4o",
            params={"temperature": 0.7},
            messages=[{"role": "user", "content": "hello"}],
            response_content="Hi there!",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=200,
        )
        snapshot = collector.finalize({})
        assert snapshot.llm_request.model == "gpt-4o"
        assert snapshot.llm_response.content == "Hi there!"
        assert snapshot.llm_response.total_tokens == 15

    def test_record_memory_ids(self):
        collector = SnapshotCollector("WritingAgent", {"name": "test"})
        collector.record_memory_ids(["mem_001", "mem_002"])
        snapshot = collector.finalize({})
        assert snapshot.context.memory_ids == ["mem_001", "mem_002"]

    def test_contextvar_set_and_get(self):
        collector = SnapshotCollector("TestAgent", {"name": "test"})
        set_collector(collector)
        try:
            retrieved = get_collector()
            assert retrieved is collector
        finally:
            set_collector(None)
        assert get_collector() is None

    def test_snapshot_has_hash(self):
        collector = SnapshotCollector("WritingAgent", {"name": "test"})
        snapshot = collector.finalize({"status": "done"})
        assert len(snapshot.captured_hash) == 12

    def test_tags_can_be_set(self):
        collector = SnapshotCollector("WritingAgent", {"name": "test"})
        collector.tags = ["test", "unit"]
        snapshot = collector.finalize({})
        assert snapshot.tags == ["test", "unit"]
