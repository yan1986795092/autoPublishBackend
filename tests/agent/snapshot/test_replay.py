import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from src.agent.snapshot.replay import ReplayEngine
from src.agent.snapshot.models import (
    ExecutionSnapshot, PromptMessageSnapshot, ContextSnapshot,
    LlmRequestSnapshot, LlmResponseSnapshot,
)
from src.agent.snapshot.storage import SnapshotStorage


@pytest.fixture
def snapshot_storage(snapshot_dir):
    return SnapshotStorage(storage_dir=str(snapshot_dir))


@pytest.fixture
def replay_engine(snapshot_storage):
    return ReplayEngine(storage=snapshot_storage)


class TestReplayEngine:
    def test_load_snapshot(self, replay_engine, sample_snapshot):
        replay_engine._storage.save(sample_snapshot)
        loaded = replay_engine.load("snap_test001")
        assert loaded is not None
        assert loaded.agent_name == "WritingAgent"

    def test_rebuild_prompt_returns_message_list(self, replay_engine, sample_snapshot):
        replay_engine._storage.save(sample_snapshot)
        messages = replay_engine.rebuild_prompt("snap_test001")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are an expert writer..."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Write a blog post about AI..."

    def test_rebuild_prompt_nonexistent(self, replay_engine):
        with pytest.raises(ValueError, match="not found"):
            replay_engine.rebuild_prompt("snap_nonexistent")

    def test_load_nonexistent(self, replay_engine):
        assert replay_engine.load("snap_nonexistent") is None

    @pytest.mark.asyncio
    async def test_rerun_generates_new_snapshot(self, replay_engine, sample_snapshot):
        replay_engine._storage.save(sample_snapshot)

        # Mock LLM
        mock_llm = AsyncMock()
        mock_llm.model_name = "gpt-4o-mini"
        mock_response = MagicMock()
        mock_response.content = "Rerun result content"
        mock_response.finish_reason = "stop"
        mock_response.usage_metadata = {}
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        new_snapshot = await replay_engine.rerun("snap_test001", llm=mock_llm)
        assert new_snapshot is not None
        assert new_snapshot.snapshot_id != "snap_test001"
        assert new_snapshot.parent_snapshot_id == "snap_test001"
        assert "Rerun" in new_snapshot.llm_response.content

    @pytest.mark.asyncio
    async def test_rerun_dry_run_no_llm(self, replay_engine, sample_snapshot):
        replay_engine._storage.save(sample_snapshot)
        new_snapshot = await replay_engine.rerun("snap_test001", llm=None)
        assert new_snapshot is not None
        assert new_snapshot.parent_snapshot_id == "snap_test001"
        assert new_snapshot.result_summary == "dry-run (no LLM call)"

    @pytest.mark.asyncio
    async def test_rerun_nonexistent(self, replay_engine):
        with pytest.raises(ValueError, match="not found"):
            await replay_engine.rerun("snap_nonexistent")
