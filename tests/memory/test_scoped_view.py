"""Tests for agent_scoped_view permission filtering."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.memory.context_store import (
    SharedContextStore, CreativeAnchors, CreativeDecision,
    ContentBlock, ReviewBlock, SEOBlock, AgentPrivateState,
    ProjectManifest, ProjectStage, AGENT_PERMISSIONS,
)
from src.memory.agent_scoped_view import (
    build_agent_context, write_to_context, log_creative_decision,
    get_context_for_writing_agent, filter_task_fields, WriteResult,
    log_review_anchor_drift,
)


class TestBuildAgentContext:
    def test_writing_agent_gets_feedback_and_seo(self):
        store = SharedContextStore()
        store.user_request = "Write a story"
        store.writing_content.append(ContentBlock(text="Once upon a time..."))
        store.review_feedback.append(ReviewBlock(text="Needs more dialogue", approved=False))
        store.seo_suggestions.append(SEOBlock(keywords=["hero"]))
        store.creative_anchors = CreativeAnchors(story_outline="Hero's journey")

        ctx = build_agent_context("WritingAgent", store)

        assert ctx["task"]["writing_content"] is not None
        assert ctx["task"]["review_feedback"] is not None
        assert ctx["task"]["seo_suggestions"] is not None
        assert ctx["task"]["creative_anchors"].story_outline == "Hero's journey"
        assert ctx["system"]["user_request"] == "Write a story"

    def test_seo_agent_cannot_read_review_feedback(self):
        store = SharedContextStore()
        store.review_feedback.append(ReviewBlock(text="Denied"))

        ctx = build_agent_context("SEOAgent", store)

        assert "review_feedback" not in ctx["task"]

    def test_review_agent_gets_creative_log(self):
        store = SharedContextStore()
        decision = CreativeDecision.create(
            decision_text="Hero changes personality",
            tags=["anchor_change"],
            agent="WritingAgent",
            turn_number=3,
        )
        store.creative_log.append(decision)

        ctx = build_agent_context("ReviewAgent", store)

        assert "creative_log" in ctx["system"]
        assert len(ctx["system"]["creative_log"]) == 1

    def test_agent_private_isolation(self):
        store = SharedContextStore()
        store.agent_private["WritingAgent"] = AgentPrivateState(confidence_score=0.95)
        store.agent_private["ReviewAgent"] = AgentPrivateState(confidence_score=0.80)

        ctx_writing = build_agent_context("WritingAgent", store)
        ctx_review = build_agent_context("ReviewAgent", store)

        assert ctx_writing["self"]["confidence_score"] == 0.95
        assert ctx_review["self"]["confidence_score"] == 0.80


class TestWriteContext:
    def test_writing_agent_appends_to_content_list(self):
        store = SharedContextStore()

        result = write_to_context("WritingAgent", "task.writing_content",
                                  ContentBlock(text="First draft"), store)
        assert result == WriteResult.ACCEPTED
        assert len(store.writing_content) == 1
        assert store.writing_content[0].text == "First draft"

        result2 = write_to_context("WritingAgent", "task.writing_content",
                                  ContentBlock(text="Second draft"), store)
        assert result2 == WriteResult.ACCEPTED
        assert len(store.writing_content) == 2
        assert store.last_writing_content.text == "Second draft"

    def test_writing_agent_cannot_write_review_feedback(self):
        store = SharedContextStore()

        result = write_to_context("WritingAgent", "task.review_feedback",
                                  ReviewBlock(text="Approved"), store)
        assert result == WriteResult.REJECTED
        assert len(store.review_feedback) == 0

    def test_seo_agent_cannot_write_content(self):
        store = SharedContextStore()

        result = write_to_context("SEOAgent", "task.writing_content",
                                  ContentBlock(text="Hijacked"), store)
        assert result == WriteResult.REJECTED

    def test_review_agent_appends_feedback(self):
        store = SharedContextStore()

        result = write_to_context("ReviewAgent", "task.review_feedback",
                                  ReviewBlock(text="First review", approved=False), store)
        assert result == WriteResult.ACCEPTED
        assert len(store.review_feedback) == 1

    def test_write_creates_thinking_step(self):
        store = SharedContextStore()
        initial_trace_len = len(store.thinking_trace)

        write_to_context("WritingAgent", "task.writing_content",
                         ContentBlock(text="Content"), store)

        assert len(store.thinking_trace) == initial_trace_len + 1
        step = store.thinking_trace[-1]
        assert step.agent == "WritingAgent"
        assert step.action == "write"


class TestLogCreativeDecision:
    def test_log_creates_decision_in_creative_log(self):
        store = SharedContextStore()
        store.project_manifest = ProjectManifest(project_id="p1", title="Test", turn_count=5)

        log_creative_decision(
            decision_text="Main character decides to leave home",
            tags=["character_arc", "inciting_incident"],
            agent="WritingAgent",
            store=store,
        )

        assert len(store.creative_log) == 1
        entry = store.creative_log[0]
        assert entry.decision_text == "Main character decides to leave home"
        assert "character_arc" in entry.tags
        assert entry.agent == "WritingAgent"
        assert entry.turn_number == 5
        assert entry.timestamp != ""

    def test_log_creates_thinking_step(self):
        store = SharedContextStore()
        store.project_manifest = ProjectManifest(project_id="p1", title="Test", turn_count=1)

        log_creative_decision(
            decision_text="No major pivot",
            tags=["continuation"],
            agent="WritingAgent",
            store=store,
        )

        step = next((s for s in store.thinking_trace if s.action == "log_creative_decision"), None)
        assert step is not None


class TestGetContextForWritingAgent:
    def test_writing_agent_gets_creative_view_with_lists(self):
        store = SharedContextStore()
        store.creative_anchors = CreativeAnchors(
            story_outline="Test outline",
            character_profiles={"protagonist": "quiet and observant"},
        )
        store.review_feedback.append(ReviewBlock(text="Good start", approved=False))
        store.seo_suggestions.append(SEOBlock(keywords=["hero", "journey"]))
        store.writing_content.append(ContentBlock(text="Chapter 1 content"))
        store.writing_content.append(ContentBlock(text="Chapter 2 content"))

        ctx = get_context_for_writing_agent(store)

        assert "creative_view" in ctx
        assert ctx["creative_view"]["anchors"].story_outline == "Test outline"
        assert ctx["creative_view"]["active_feedback"]["review"].text == "Good start"
        assert ctx["creative_view"]["active_feedback"]["seo"].keywords == ["hero", "journey"]
        # recent_content returns last 3
        assert len(ctx["creative_view"]["recent_content"]) == 2
        assert ctx["creative_view"]["recent_content"][-1].text == "Chapter 2 content"


class TestFilterTaskFields:
    def test_writing_agent_filters_correctly(self):
        store = SharedContextStore()
        store.writing_content.append(ContentBlock(text="content"))
        store.review_feedback.append(ReviewBlock(text="feedback"))

        result = filter_task_fields("WritingAgent", store)

        assert "writing_content" in result
        assert "review_feedback" in result

    def test_seo_agent_filters_correctly(self):
        store = SharedContextStore()
        store.review_feedback.append(ReviewBlock(text="feedback"))

        result = filter_task_fields("SEOAgent", store)

        assert "review_feedback" not in result


class TestLogReviewAnchorDrift:
    def test_log_review_anchor_drift_creates_entry(self):
        store = SharedContextStore()
        store.project_manifest = ProjectManifest(project_id="p1", title="Test", turn_count=2)

        log_review_anchor_drift(
            drift_text="character_profiles.hero: contradictory trait detected",
            severity="high",
            store=store,
        )

        assert len(store.creative_log) == 1
        entry = store.creative_log[0]
        assert "review_anchor_drift" in entry.tags
        assert entry.agent == "ReviewAgent"
