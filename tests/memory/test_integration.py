"""End-to-end integration tests for multi-agent context management."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.memory import (
    ContextManager,
    SharedContextStore,
    ConsolidatorNode,
    build_agent_context,
    write_to_context,
    log_creative_decision,
    get_context_for_writing_agent,
    CreativeAnchors,
    ContentBlock,
    ReviewBlock,
    SEOBlock,
    ProjectManifest,
    ProjectStage,
    WriteResult,
    EntityMemory,
    PlotMemory,
    ProjectIndex,
    CreativeSummaryMemory,
    AnchorConsistencyChecker,
    StructuredMemoryEntry,
)


class TestFullContextLifecycle:
    def test_new_conversation_creates_store(self):
        cm = ContextManager()
        store = cm.handle_message("Write me a fantasy novel", "conv-1", None)
        assert store.is_new_session
        assert store.project_manifest.stage == ProjectStage.CONCEPTUALIZATION

    def test_continue_session_preserves_anchors(self):
        cm = ContextManager()
        msg = "Write me a fantasy novel about hero"
        store1 = cm.handle_message(msg, "conv-1", None)
        store1.creative_anchors = CreativeAnchors(
            story_outline="Hero leaves home",
            character_profiles={"hero": "brave"},
        )
        store2 = cm.handle_message(msg, "conv-1", None)
        assert not store2.is_new_session
        assert store2.creative_anchors.story_outline == "Hero leaves home"
        assert store2.project_manifest.turn_count == 1

    def test_writing_agent_self_corrects_from_feedback(self):
        cm = ContextManager()
        store = cm.handle_message("Write a story about hero", "conv-1", None)

        write_to_context("WritingAgent", "task.writing_content",
                        ContentBlock(text="The hero was quiet."), store)
        assert store.writing_content[0].text == "The hero was quiet."

        write_to_context("ReviewAgent", "task.review_feedback",
                        ReviewBlock(text="Add more dialogue"), store)

        ctx = get_context_for_writing_agent(store)
        assert ctx["creative_view"]["active_feedback"]["review"].text == "Add more dialogue"

        write_to_context("WritingAgent", "task.writing_content",
                        ContentBlock(text="The hero spoke: 'I will go.'"), store)
        assert "I will go" in store.last_writing_content.text

        result = write_to_context("SEOAgent", "task.writing_content",
                                 ContentBlock(text="SEO hijack"), store)
        assert result == WriteResult.REJECTED

    def test_creative_decisions_are_preserved(self):
        cm = ContextManager()
        store = cm.handle_message("Write a story about hero adventure", "conv-1", None)
        store.project_manifest.turn_count = 3

        log_creative_decision(
            decision_text="Hero chooses the left path",
            tags=["plot_decision"],
            agent="WritingAgent",
            store=store,
        )

        assert len(store.creative_log) == 1
        assert "left path" in store.creative_log[0].decision_text
        assert "plot_decision" in store.creative_log[0].tags

    def test_consolidator_writes_memory_on_completion(self):
        cm = ContextManager()
        store = cm.handle_message("Write a chapter hero adventure", "conv-1", None)
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test Novel", stage=ProjectStage.COMPLETED,
        )
        store.writing_content.append(ContentBlock(text="Chapter 1 complete."))

        consolidator = ConsolidatorNode()
        result = consolidator.finalize(store)

        assert result.status == "completed"

    def test_stage_transition_validation(self):
        cm = ContextManager()
        store = cm.handle_message("Write a novel hero adventure", "conv-1", None)
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test", stage=ProjectStage.DRAFTING,
        )

        ok = cm.transition_stage(store, ProjectStage.REVISION)
        assert ok
        assert store.project_manifest.stage == ProjectStage.REVISION

        store.project_manifest.stage = ProjectStage.COMPLETED
        ok = cm.transition_stage(store, ProjectStage.DRAFTING)
        assert not ok
        assert store.project_manifest.stage == ProjectStage.COMPLETED

    def test_creative_anchors_never_compressed(self):
        cm = ContextManager()
        store = cm.handle_message("Write novel hero adventure", "conv-1", None)
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test", stage=ProjectStage.DRAFTING,
            turn_count=20,
        )
        store.creative_anchors = CreativeAnchors(
            story_outline="Hero leaves home",
            character_profiles={"hero": "brave"},
            world_setting="Medieval fantasy",
            style_guide="Terse prose.",
            plot_milestones=["inciting", "midpoint"],
        )

        consolidator = ConsolidatorNode()
        consolidator._compress(store)

        assert store.creative_anchors.story_outline == "Hero leaves home"
        assert store.creative_anchors.character_profiles["hero"] == "brave"
        assert store.creative_anchors.world_setting == "Medieval fantasy"

    def test_structured_memory_integration(self):
        em = EntityMemory()
        pm = PlotMemory()
        idx = ProjectIndex()
        csm = CreativeSummaryMemory()

        idx.register("novel-1", "The Hero's Journey", "novel")
        idx.update_stage("novel-1", ProjectStage.DRAFTING)

        em.store(StructuredMemoryEntry(
            entry_type="entity",
            entity_id="novel-1:hero",
            project_id="novel-1",
            attributes={"name": "Hero", "trait": "brave"},
            vector_summary="The main hero character.",
        ))

        pm.store(StructuredMemoryEntry(
            entry_type="plot_event",
            entity_id="evt-1",
            project_id="novel-1",
            attributes={"event": "Hero leaves home", "turn": "1"},
            vector_summary="Inciting incident.",
        ))

        csm.store(
            project_id="novel-1",
            intent="Hero's journey",
            output_summary="Chapter 1: Hero leaves home.",
            style_template="Terse prose.",
        )

        assert idx.get("novel-1")["stage"] == "drafting"
        assert len(em.retrieve_by_project("novel-1")) == 1
        assert len(pm.retrieve_by_project("novel-1")) == 1
        assert csm.get_latest("novel-1") is not None

    def test_permission_model_all_agents(self):
        store = SharedContextStore()
        store.writing_content.append(ContentBlock(text="content"))
        store.review_feedback.append(ReviewBlock(text="feedback"))
        store.seo_suggestions.append(SEOBlock(keywords=["hero"]))

        # WritingAgent can read all three
        ctx_w = build_agent_context("WritingAgent", store)
        assert "writing_content" in ctx_w["task"]
        assert "review_feedback" in ctx_w["task"]
        assert "seo_suggestions" in ctx_w["task"]

        # ReviewAgent cannot write content
        result = write_to_context("ReviewAgent", "task.writing_content",
                                 ContentBlock(text="hijacked"), store)
        assert result == WriteResult.REJECTED

        # SEOAgent cannot write content or review
        result = write_to_context("SEOAgent", "task.writing_content",
                                 ContentBlock(text="hijacked"), store)
        assert result == WriteResult.REJECTED
        result = write_to_context("SEOAgent", "task.review_feedback",
                                 ReviewBlock(text="Approved"), store)
        assert result == WriteResult.REJECTED

    def test_anchor_drift_detection(self):
        checker = AnchorConsistencyChecker()
        anchors = CreativeAnchors(character_profiles={"hero": "introverted"})
        content = "The hero was loud and extroverted today."

        drifts = checker.check(anchors, content)
        assert len(drifts) > 0
        assert drifts[0].drift_type == "contradiction"
        assert drifts[0].anchor_field == "character_profiles.hero"
