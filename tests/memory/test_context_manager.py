"""Tests for ContextManager — hybrid routing, session management, compression."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.memory.context_store import (
    SharedContextStore, CreativeAnchors, ProjectManifest, ProjectStage, ThinkingStep,
)
from src.memory.context_manager import ContextManager


class TestContextManagerRouting:
    def test_new_user_request_creates_store(self):
        manager = ContextManager()

        store = manager.handle_message(
            user_message="Write me a fantasy novel",
            conversation_id="conv-1",
            project_id=None,
        )

        assert store is not None
        assert store.is_new_session is True
        assert store.user_request == "Write me a fantasy novel"
        assert store.conversation_id == "conv-1"
        assert store.session_id != ""

    def test_continue_session_within_threshold(self):
        manager = ContextManager()
        msg = "Write me a fantasy story about hero adventure"
        store1 = manager.handle_message(msg, "conv-1", None)
        store1.creative_anchors = CreativeAnchors(story_outline="Hero's journey")

        # Same exact message guarantees keyword similarity = 1.0 (exceeds 0.75 threshold)
        store2 = manager.handle_message(msg, "conv-1", None)

        assert store2.is_new_session is False
        assert store2.session_id == store1.session_id
        assert store2.creative_anchors.story_outline == "Hero's journey"

    def test_different_conversation_id_creates_new_session(self):
        manager = ContextManager()
        store1 = manager.handle_message("Write me a fantasy novel", "conv-1", None)

        store2 = manager.handle_message("Completely different task", "conv-2", None)

        assert store2.is_new_session is True
        assert store2.session_id != store1.session_id

    def test_explicit_new_session_directive(self):
        manager = ContextManager()
        store1 = manager.handle_message("Write me a fantasy novel", "conv-1", None)

        store2 = manager.handle_message(
            "Start a new story about a detective",
            "conv-1",
            project_directive="new_session",
        )

        assert store2.is_new_session is True
        assert store2.session_id != store1.session_id

    def test_turn_count_increments_on_continue(self):
        manager = ContextManager()
        msg = "Write me a story about hero adventure"
        store1 = manager.handle_message(msg, "conv-1", None)

        # Same message -> similarity = 1.0 -> continues session -> turn_count increments
        store2 = manager.handle_message(msg, "conv-1", None)

        assert store2.project_manifest.turn_count == 1

    def test_archive_on_new_session(self):
        manager = ContextManager()
        store1 = manager.handle_message("First task", "conv-1", None)
        session_id_1 = store1.session_id

        store2 = manager.handle_message("Second task", "conv-1", None)

        assert store2.session_id != session_id_1
        history = manager.get_session_history()
        assert any(s.session_id == session_id_1 for s in history)


class TestStageTransitions:
    def test_valid_transition(self):
        manager = ContextManager()
        store = manager.handle_message("Write me a novel", "conv-1", None)
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Novel", stage=ProjectStage.DRAFTING,
        )

        result = manager.transition_stage(store, ProjectStage.REVISION)

        assert result is True
        assert store.project_manifest.stage == ProjectStage.REVISION

    def test_invalid_transition_rejected(self):
        manager = ContextManager()
        store = manager.handle_message("Write me a novel", "conv-1", None)
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Novel", stage=ProjectStage.COMPLETED,
        )

        result = manager.transition_stage(store, ProjectStage.DRAFTING)

        assert result is False
        assert store.project_manifest.stage == ProjectStage.COMPLETED

    def test_backwards_transition_allowed(self):
        manager = ContextManager()
        store = manager.handle_message("Write me a novel", "conv-1", None)
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Novel", stage=ProjectStage.REVISION,
        )

        result = manager.transition_stage(store, ProjectStage.DRAFTING)

        assert result is True
        assert store.project_manifest.stage == ProjectStage.DRAFTING


class TestTokenBudget:
    def test_estimate_tokens_returns_positive_int(self):
        manager = ContextManager()

        store = SharedContextStore()
        store.user_request = "A" * 1000
        store.creative_anchors = CreativeAnchors(story_outline="X" * 500)

        tokens = manager.estimate_token_count(store)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_compression_preserves_creative_anchors(self):
        manager = ContextManager()
        store = SharedContextStore()
        store.user_request = "Write me a novel"
        store.creative_anchors = CreativeAnchors(
            story_outline="Hero leaves home",
            character_profiles={"hero": "brave"},
        )
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test", stage=ProjectStage.DRAFTING,
            turn_count=20,
        )
        for i in range(50):
            store.thinking_trace.append(ThinkingStep(
                agent="WritingAgent", action="reasoning", detail=f"step {i}",
            ))

        manager.compress_context(store)

        assert store.creative_anchors.story_outline == "Hero leaves home"
        assert store.creative_anchors.character_profiles["hero"] == "brave"

    def test_compression_trims_thinking_trace(self):
        manager = ContextManager()
        store = SharedContextStore()
        store.user_request = "Write"
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test", stage=ProjectStage.DRAFTING, turn_count=20,
        )
        for i in range(50):
            store.thinking_trace.append(ThinkingStep(
                agent="WritingAgent", action="reasoning", detail=f"step {i}",
            ))

        manager.compress_context(store)

        # Compression trims 50 down to last 20, then adds compress_start + compress_end
        assert 20 <= len(store.thinking_trace) <= 22
        # The last two steps should be the compression markers
        assert store.thinking_trace[-1].action == "compress_end"
        assert store.thinking_trace[-2].action == "compress_start"

    def test_should_compress_above_threshold(self):
        manager = ContextManager()
        store = SharedContextStore()
        # 102401 chars / 4 = 25601 tokens >= 80% of 32000 budget
        store.user_request = "Write" + ("x" * 102401)
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test", stage=ProjectStage.DRAFTING, turn_count=20,
        )

        should = manager.should_compress(store, threshold_pct=0.80)
        assert should is True

    def test_should_compress_below_threshold(self):
        manager = ContextManager()
        store = SharedContextStore()
        store.user_request = "Short prompt"

        should = manager.should_compress(store, threshold_pct=0.80)
        assert should is False


class TestSessionManagement:
    def test_clear_session_removes_active(self):
        manager = ContextManager()
        store = manager.handle_message("Write a story", "conv-1", None)
        session_id = store.session_id

        manager.clear_session(session_id)

        assert manager.get_active_session("conv-1") is None

    def test_get_active_session_returns_store(self):
        manager = ContextManager()
        store = manager.handle_message("Write a story", "conv-1", None)

        result = manager.get_active_session("conv-1")
        assert result is not None
        assert result.session_id == store.session_id
