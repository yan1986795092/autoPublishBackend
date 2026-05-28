"""Tests for Consolidator — session finalization and anchor checking."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.memory.context_store import (
    SharedContextStore, CreativeAnchors, ProjectManifest, ProjectStage,
    ContentBlock, StructuredMemoryEntry,
)
from src.memory.consolidator import ConsolidatorNode, AnchorConsistencyChecker


class TestAnchorConsistencyChecker:
    def test_no_drift_when_content_matches_anchor(self):
        checker = AnchorConsistencyChecker()
        anchors = CreativeAnchors(
            character_profiles={"hero": "brave and kind"},
            style_guide="Terse prose.",
        )
        content = "The hero walked into the cave, brave and kind as ever."

        drifts = checker.check(anchors, content)
        high_severity = [d for d in drifts if d.severity == "high"]
        assert len(high_severity) == 0

    def test_detects_explicit_trait_contradiction(self):
        checker = AnchorConsistencyChecker()
        anchors = CreativeAnchors(
            character_profiles={"hero": "introverted"},
        )
        content = "The hero shouted loudly, an extroverted personality shining through."

        drifts = checker.check(anchors, content)
        contradictions = [d for d in drifts if d.drift_type == "contradiction"]
        assert len(contradictions) > 0
        assert contradictions[0].anchor_field == "character_profiles.hero"

    def test_empty_content_returns_no_drifts(self):
        checker = AnchorConsistencyChecker()
        anchors = CreativeAnchors(character_profiles={"hero": "brave"})

        drifts = checker.check(anchors, "")
        assert len(drifts) == 0

    def test_no_anchors_returns_no_drifts(self):
        checker = AnchorConsistencyChecker()
        anchors = CreativeAnchors()
        content = "Something about a brave hero."

        drifts = checker.check(anchors, content)
        assert len(drifts) == 0


class TestConsolidatorNode:
    def test_finalize_completed_status_when_content_exists(self):
        node = ConsolidatorNode()
        store = SharedContextStore()
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test Novel", stage=ProjectStage.COMPLETED,
        )
        store.writing_content.append(ContentBlock(text="Chapter 1 complete."))

        result = node.finalize(store)

        assert result.status == "completed"

    def test_finalize_failed_status_when_no_content(self):
        node = ConsolidatorNode()
        store = SharedContextStore()
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test", stage=ProjectStage.COMPLETED,
        )

        result = node.finalize(store)

        assert result.status == "failed"

    def test_finalize_adds_thinking_step(self):
        node = ConsolidatorNode()
        store = SharedContextStore()
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test", stage=ProjectStage.COMPLETED,
        )

        initial_len = len(store.thinking_trace)
        node.finalize(store)

        assert len(store.thinking_trace) > initial_len

    def test_finalize_detects_anchor_drifts(self):
        node = ConsolidatorNode()
        store = SharedContextStore()
        store.creative_anchors = CreativeAnchors(
            character_profiles={"hero": "introverted"},
        )
        store.writing_content.append(ContentBlock(
            text="The hero was loud and extroverted today.",
        ))
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test", stage=ProjectStage.COMPLETED,
        )

        result = node.finalize(store)

        assert len(result.anchor_drifts) > 0

    def test_finalize_generates_summary(self):
        node = ConsolidatorNode()
        store = SharedContextStore()
        store.project_manifest = ProjectManifest(
            project_id="p1", title="The Hero's Journey", stage=ProjectStage.COMPLETED,
            turn_count=10,
        )
        store.writing_content.append(ContentBlock(text="Chapter 1."))

        result = node.finalize(store)

        assert result.project_summary != ""
        assert "The Hero's Journey" in result.project_summary

    def test_compress_threshold_respected(self):
        node = ConsolidatorNode(compress_threshold_pct=0.0)
        store = SharedContextStore()
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test", stage=ProjectStage.COMPLETED,
        )
        from src.memory.context_store import ThinkingStep
        for i in range(30):
            store.thinking_trace.append(ThinkingStep(
                agent="WritingAgent", action="reasoning", detail=f"step {i}",
            ))

        result = node.finalize(store)

        # Consolidator should have triggered compression
        assert len(store.thinking_trace) <= 30
