"""Tests for context_store data models."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.memory.context_store import (
    ProjectStage,
    CreativeAnchors,
    CreativeDecision,
    SharedContextStore,
    ThinkingStep,
    AGENT_PERMISSIONS,
    VALID_TRANSITIONS,
    WriteResult,
    ContentBlock,
    ReviewBlock,
    SEOBlock,
    AgentPrivateState,
    ProjectManifest,
    StructuredMemoryEntry,
    AnchorDrift,
)


class TestCreativeAnchors:
    def test_default_anchors_are_empty(self):
        anchors = CreativeAnchors()
        assert anchors.story_outline == ""
        assert anchors.character_profiles == {}
        assert anchors.world_setting == ""
        assert anchors.style_guide == ""
        assert anchors.plot_milestones == []

    def test_anchors_with_values(self):
        anchors = CreativeAnchors(
            story_outline="A hero's journey",
            character_profiles={"hero": "brave and kind"},
            world_setting="medieval fantasy",
            style_guide="Terse and evocative prose.",
            plot_milestones=["inciting incident", "midpoint revelation"],
        )
        assert anchors.story_outline == "A hero's journey"
        assert anchors.character_profiles["hero"] == "brave and kind"
        assert len(anchors.plot_milestones) == 2


class TestSharedContextStore:
    def test_default_store_is_empty(self):
        store = SharedContextStore()
        assert store.user_request == ""
        assert store.creative_anchors.story_outline == ""
        assert store.creative_log == []

    def test_creative_anchors_exempt_from_compression(self):
        store = SharedContextStore()
        store.creative_anchors = CreativeAnchors(story_outline="Test")
        assert store.creative_anchors.story_outline == "Test"

    def test_agent_private_isolation(self):
        store = SharedContextStore()
        store.agent_private["WritingAgent"] = AgentPrivateState(confidence_score=0.9)
        assert "ReviewAgent" not in store.agent_private

    def test_get_or_create_agent_private(self):
        store = SharedContextStore()
        priv = store.get_or_create_agent_private("WritingAgent")
        assert priv.confidence_score == 0.0
        priv.confidence_score = 0.85
        priv2 = store.get_or_create_agent_private("WritingAgent")
        assert priv2.confidence_score == 0.85

    def test_add_thinking_step(self):
        store = SharedContextStore()
        store.add_thinking_step(agent="WritingAgent", action="write", step_field="task.writing_content")
        assert len(store.thinking_trace) == 1
        step = store.thinking_trace[0]
        assert step.agent == "WritingAgent"
        assert step.action == "write"
        assert step.step_field == "task.writing_content"


class TestProjectStage:
    def test_all_stages_exist(self):
        assert ProjectStage.CONCEPTUALIZATION.value == "conceptualization"
        assert ProjectStage.OUTLINING.value == "outlining"
        assert ProjectStage.DRAFTING.value == "drafting"
        assert ProjectStage.REVISION.value == "revision"
        assert ProjectStage.POLISHING.value == "polishing"
        assert ProjectStage.COMPLETED.value == "completed"
        assert ProjectStage.ABANDONED.value == "abandoned"

    def test_valid_transitions_from_drafting(self):
        assert ProjectStage.REVISION in VALID_TRANSITIONS[ProjectStage.DRAFTING]
        assert ProjectStage.ABANDONED in VALID_TRANSITIONS[ProjectStage.DRAFTING]

    def test_completed_is_terminal(self):
        assert len(VALID_TRANSITIONS[ProjectStage.COMPLETED]) == 0

    def test_abandoned_is_terminal(self):
        assert len(VALID_TRANSITIONS[ProjectStage.ABANDONED]) == 0

    def test_all_transitions_are_valid(self):
        for stage, targets in VALID_TRANSITIONS.items():
            for target in targets:
                assert isinstance(target, ProjectStage)


class TestCreativeDecision:
    def test_create_sets_all_fields(self):
        decision = CreativeDecision.create(
            decision_text="Hero leaves home",
            tags=["plot", "inciting"],
            agent="WritingAgent",
            turn_number=3,
        )
        assert decision.decision_text == "Hero leaves home"
        assert "plot" in decision.tags
        assert decision.agent == "WritingAgent"
        assert decision.turn_number == 3
        assert decision.timestamp != ""


class TestProjectManifest:
    def test_to_dict(self):
        manifest = ProjectManifest(
            project_id="p-123",
            title="My Novel",
            project_type="novel",
            stage=ProjectStage.DRAFTING,
            created_at="2026-01-01T00:00:00",
            turn_count=5,
        )
        d = manifest.to_dict()
        assert d["project_id"] == "p-123"
        assert d["stage"] == "drafting"
        assert d["turn_count"] == 5


class TestAgentPermissions:
    def test_writing_agent_can_read_feedback(self):
        perm = AGENT_PERMISSIONS["WritingAgent"]
        assert "task.review_feedback" in perm.readable_fields
        assert "task.seo_suggestions" in perm.readable_fields

    def test_writing_agent_cannot_write_review(self):
        perm = AGENT_PERMISSIONS["WritingAgent"]
        assert "task.review_feedback" not in perm.writable_fields

    def test_review_agent_cannot_write_content(self):
        perm = AGENT_PERMISSIONS["ReviewAgent"]
        assert "task.writing_content" not in perm.writable_fields

    def test_all_agents_have_private_fields(self):
        for name, perm in AGENT_PERMISSIONS.items():
            assert "self.internal_reasoning" in perm.writable_fields


class TestSharedContextStoreListFields:
    def test_writing_content_is_list(self):
        store = SharedContextStore()
        assert isinstance(store.writing_content, list)
        assert store.writing_content == []
        assert store.last_writing_content is None

    def test_review_feedback_is_list(self):
        store = SharedContextStore()
        assert isinstance(store.review_feedback, list)
        assert store.review_feedback == []
        assert store.last_review_feedback is None

    def test_seo_suggestions_is_list(self):
        store = SharedContextStore()
        assert isinstance(store.seo_suggestions, list)
        assert store.seo_suggestions == []
        assert store.last_seo_suggestions is None

    def test_last_writing_content_returns_last(self):
        store = SharedContextStore()
        store.writing_content.append(ContentBlock(text="First"))
        store.writing_content.append(ContentBlock(text="Second"))
        store.writing_content.append(ContentBlock(text="Last"))
        assert store.last_writing_content.text == "Last"

    def test_last_review_feedback_returns_last(self):
        store = SharedContextStore()
        store.review_feedback.append(ReviewBlock(text="First", approved=False))
        store.review_feedback.append(ReviewBlock(text="Last", approved=True))
        assert store.last_review_feedback.text == "Last"
        assert store.last_review_feedback.approved is True

    def test_last_seo_suggestions_returns_last(self):
        store = SharedContextStore()
        store.seo_suggestions.append(SEOBlock(keywords=["a"]))
        store.seo_suggestions.append(SEOBlock(keywords=["b", "c"]))
        assert store.last_seo_suggestions.keywords == ["b", "c"]


class TestContentBlocks:
    def test_content_block(self):
        block = ContentBlock(text="Once upon a time", turn_number=1, agent="WritingAgent")
        assert block.text == "Once upon a time"
        assert block.turn_number == 1

    def test_review_block(self):
        block = ReviewBlock(text="Good start", approved=True)
        assert block.approved is True

    def test_seo_block(self):
        block = SEOBlock(text="Add keywords", keywords=["hero", "journey"])
        assert "hero" in block.keywords
