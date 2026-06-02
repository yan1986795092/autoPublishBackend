"""SnapshotStorage 单元测试"""
import json
import pytest
from pathlib import Path
from datetime import datetime

from src.agent.snapshot.models import ExecutionSnapshot
from src.agent.snapshot.storage import SnapshotStorage


class TestSnapshotStorage:
    def test_save_and_load(self, snapshot_dir, sample_snapshot):
        storage = SnapshotStorage(storage_dir=str(snapshot_dir))
        storage.save(sample_snapshot)

        loaded = storage.load(sample_snapshot.snapshot_id)
        assert loaded is not None
        assert loaded.snapshot_id == "snap_test001"
        assert loaded.agent_name == "WritingAgent"
        assert len(loaded.prompt_messages) == 2
        assert loaded.prompt_messages[0].template_name == "system.base"
        assert loaded.llm_response.total_tokens == 700

    def test_load_nonexistent(self, snapshot_dir):
        storage = SnapshotStorage(storage_dir=str(snapshot_dir))
        assert storage.load("snap_nonexistent") is None

    def test_save_many_and_list(self, snapshot_dir, sample_snapshot):
        storage = SnapshotStorage(storage_dir=str(snapshot_dir))
        storage.save(sample_snapshot)

        snap2 = ExecutionSnapshot(
            snapshot_id="snap_test002",
            agent_name="ReviewAgent",
            timestamp=sample_snapshot.timestamp,
        )
        storage.save(snap2)

        all_snaps = storage.list(limit=10)
        assert len(all_snaps) == 2
        assert all_snaps[0].snapshot_id == "snap_test002"

    def test_list_with_agent_filter(self, snapshot_dir):
        from datetime import datetime
        storage = SnapshotStorage(storage_dir=str(snapshot_dir))
        ts = datetime(2026, 6, 1, 12, 0, 0)
        for i in range(3):
            storage.save(ExecutionSnapshot(
                snapshot_id=f"snap_w{i:03d}", agent_name="WritingAgent", timestamp=ts,
            ))
        for i in range(2):
            storage.save(ExecutionSnapshot(
                snapshot_id=f"snap_r{i:03d}", agent_name="ReviewAgent", timestamp=ts,
            ))

        writing = storage.list(agent_name="WritingAgent")
        assert len(writing) == 3

        review = storage.list(agent_name="ReviewAgent")
        assert len(review) == 2

    def test_save_roundtrip_preserves_tags(self, snapshot_dir, sample_snapshot):
        storage = SnapshotStorage(storage_dir=str(snapshot_dir))
        storage.save(sample_snapshot)
        loaded = storage.load("snap_test001")
        assert loaded.tags == ["test", "integration"]

    def test_save_creates_json_file(self, snapshot_dir, sample_snapshot):
        storage = SnapshotStorage(storage_dir=str(snapshot_dir))
        storage.save(sample_snapshot)
        json_path = snapshot_dir / "202606" / "snap_test001.json"
        assert json_path.exists()
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["snapshot_id"] == "snap_test001"
