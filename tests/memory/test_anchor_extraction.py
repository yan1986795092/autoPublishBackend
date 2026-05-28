"""Tests for anchor extraction, update, and drift detection functions."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.memory.context_store import (
    SharedContextStore,
    CreativeAnchors,
    ProjectManifest,
    ProjectStage,
)
from src.memory.agent_scoped_view import (
    extract_anchors_from_content,
    update_creative_anchors,
    check_anchor_drift_from_review,
)


class TestExtractAnchorsFromContent:
    def test_extracts_story_outline(self):
        content = """# Outline

The hero leaves his village and finds adventure.
"""
        result = extract_anchors_from_content(content)
        assert "hero" in result["story_outline"].lower() or "leaves" in result["story_outline"].lower()

    def test_extracts_character_profiles(self):
        content = """## Characters

- Alice: brave and kind-hearted
- Bob: introverted and mysterious
"""
        result = extract_anchors_from_content(content)
        assert "Alice" in result["character_profiles"]
        assert result["character_profiles"]["Alice"] == "brave and kind-hearted"
        assert "Bob" in result["character_profiles"]

    def test_extracts_plot_milestones(self):
        content = """### Plot Milestones

- Inciting Incident
- Midpoint Revelation
- Climax
"""
        result = extract_anchors_from_content(content)
        assert "Inciting Incident" in result["plot_milestones"]
        assert "Midpoint Revelation" in result["plot_milestones"]

    def test_extracts_style_guide(self):
        content = """## Style

Terse and evocative prose. Short sentences preferred.
"""
        result = extract_anchors_from_content(content)
        assert "Terse" in result["style_guide"] or "short" in result["style_guide"].lower()

    def test_extracts_world_setting(self):
        content = """### World Setting

Medieval fantasy realm with magic and dragons.
"""
        result = extract_anchors_from_content(content)
        assert "Medieval" in result["world_setting"] or "fantasy" in result["world_setting"]

    def test_empty_content_returns_empty_dict(self):
        result = extract_anchors_from_content("")
        assert result["story_outline"] == ""
        assert result["character_profiles"] == {}

    def test_no_markers_returns_empty(self):
        content = "Just some random prose without any structural markers."
        result = extract_anchors_from_content(content)
        assert result["story_outline"] == ""
        assert result["character_profiles"] == {}


class TestUpdateCreativeAnchors:
    def test_updates_empty_anchors_from_content(self):
        store = SharedContextStore()
        store.creative_anchors = CreativeAnchors()

        result = {
            "output": {
                "content": """# Outline

A brave hero travels the world.

## Characters
- Kira: quiet and observant, master of stealth

### Plot Milestones
- inciting incident
- midpoint revelation

## Style
Evocative and poetic prose.
"""
            }
        }

        updated = update_creative_anchors(result, store)

        assert updated
        assert store.creative_anchors.story_outline != ""
        assert "Kira" in store.creative_anchors.character_profiles
        assert len(store.creative_anchors.plot_milestones) > 0
        assert store.creative_anchors.style_guide != ""

    def test_does_not_overwrite_existing_anchors(self):
        store = SharedContextStore()
        store.creative_anchors = CreativeAnchors(
            story_outline="Existing outline",
            character_profiles={"Hero": "existing profile"},
        )

        result = {
            "output": {
                "content": "# Outline\n\nNew outline here.\n\n## Characters\n- NewChar: new profile"
            }
        }

        updated = update_creative_anchors(result, store)

        # Nothing updated since both fields were already set
        assert not updated
        assert store.creative_anchors.story_outline == "Existing outline"
        assert "Hero" in store.creative_anchors.character_profiles
        assert "NewChar" not in store.creative_anchors.character_profiles

    def test_returns_false_for_empty_content(self):
        store = SharedContextStore()
        result = {"output": {"content": ""}}
        updated = update_creative_anchors(result, store)
        assert not updated

    def test_creates_thinking_step_on_update(self):
        store = SharedContextStore()
        initial_len = len(store.thinking_trace)

        update_creative_anchors({
            "output": {
                "content": "# Outline\n\nA story."
            }
        }, store)

        assert len(store.thinking_trace) > initial_len
        step = store.thinking_trace[-1]
        assert step.agent == "WritingAgent"
        assert step.action == "anchor_update"

    def test_handles_string_output(self):
        store = SharedContextStore()
        result = "# Outline\n\nDirect string output."
        updated = update_creative_anchors(result, store)
        assert updated


class TestCheckAnchorDriftFromReview:
    def test_detects_drift_keyword(self):
        store = SharedContextStore()
        store.project_manifest = ProjectManifest(project_id="p1", title="Test", turn_count=2)

        result = {
            "output": {
                "content": "The content contradicts the character anchor settings.",
                "suggestions": ["Fix the inconsistency issue"],
            }
        }

        drifts = check_anchor_drift_from_review(result, store)

        assert drifts >= 1
        assert len(store.creative_log) >= 1
        entry = store.creative_log[-1]
        assert entry.tags == ['review_anchor_drift', 'severity:medium']

    def test_no_drift_on_clean_review(self):
        store = SharedContextStore()
        result = {
            "output": {
                "content": "Content looks great. Approved.",
                "suggestions": [],
            }
        }

        drifts = check_anchor_drift_from_review(result, store)
        assert drifts == 0

    def test_detects_drift_in_suggestions(self):
        store = SharedContextStore()
        result = {
            "output": {
                "content": "Some minor issues found.",
                "suggestions": ["Address the inconsistency in character behavior"],
            }
        }

        drifts = check_anchor_drift_from_review(result, store)
        assert drifts >= 1

    def test_no_exception_on_empty_output(self):
        store = SharedContextStore()
        drifts = check_anchor_drift_from_review({"output": {}}, store)
        assert drifts == 0


class TestCompressionIntegration:
    def test_compress_respects_stage_polishing(self):
        from src.memory import ContextManager, ContentBlock, ThinkingStep

        cm = ContextManager()
        store = cm.handle_message("Write novel", "conv-x", None)
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test", stage=ProjectStage.POLISHING,
        )

        for i in range(30):
            store.thinking_trace.append(ThinkingStep(
                agent="WritingAgent", action="step", detail=f"step {i}",
            ))
        for i in range(10):
            store.writing_content.append(ContentBlock(text=f"Content block {i}", agent="WritingAgent"))

        cm.compress_context(store)

        # Polishing: thinking_trace capped at 10, writing_content capped at 2
        # compression + 2 markers; handle_message adds ~8 initial steps
        assert 8 <= len(store.thinking_trace) <= 12
        assert len(store.writing_content) <= 2

    def test_compress_preserves_all_content_in_conceptualization(self):
        from src.memory import ContextManager, ContentBlock, ThinkingStep

        cm = ContextManager()
        store = cm.handle_message("Write novel", "conv-x", None)
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test", stage=ProjectStage.CONCEPTUALIZATION,
        )

        for i in range(50):
            store.thinking_trace.append(ThinkingStep(
                agent="WritingAgent", action="step", detail=f"step {i}",
            ))
        for i in range(20):
            store.writing_content.append(ContentBlock(text=f"Content {i}", agent="WritingAgent"))

        cm.compress_context(store)

        # Conceptualization: all writing_content kept; thinking_trace capped at 30 + 2 compress markers
        assert len(store.writing_content) == 20
        assert len(store.thinking_trace) <= 32

    def test_compress_never_touches_creative_anchors(self):
        from src.memory import ContextManager, ContentBlock

        cm = ContextManager()
        store = cm.handle_message("Write novel", "conv-x", None)
        store.creative_anchors = CreativeAnchors(
            story_outline="Hero's epic journey",
            character_profiles={"Hero": "brave"},
            world_setting="Fantasy world",
            style_guide="Terse prose",
            plot_milestones=["start", "mid", "end"],
        )
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test", stage=ProjectStage.POLISHING,
        )

        cm.compress_context(store)

        assert store.creative_anchors.story_outline == "Hero's epic journey"
        assert store.creative_anchors.character_profiles["Hero"] == "brave"
        assert len(store.creative_anchors.plot_milestones) == 3

    def test_compress_never_touches_creative_log(self):
        from src.memory import ContextManager, CreativeDecision, ContentBlock

        cm = ContextManager()
        store = cm.handle_message("Write novel", "conv-x", None)

        for i in range(100):
            store.creative_log.append(CreativeDecision.create(
                decision_text=f"Decision {i}",
                tags=["test"],
                agent="WritingAgent",
                turn_number=i,
            ))

        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test", stage=ProjectStage.POLISHING,
        )

        cm.compress_context(store)

        assert len(store.creative_log) == 100

    def test_compress_trims_review_feedback_to_last_2(self):
        from src.memory import ContextManager, ReviewBlock

        cm = ContextManager()
        store = cm.handle_message("Write novel", "conv-x", None)
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test", stage=ProjectStage.DRAFTING,
        )

        for i in range(5):
            store.review_feedback.append(ReviewBlock(text=f"Review {i}", approved=i > 3))

        cm.compress_context(store)

        assert len(store.review_feedback) == 2
        assert store.review_feedback[-1].text == "Review 4"

    def test_compress_trims_seo_suggestions_to_last_1(self):
        from src.memory import ContextManager, SEOBlock

        cm = ContextManager()
        store = cm.handle_message("Write novel", "conv-x", None)
        store.project_manifest = ProjectManifest(
            project_id="p1", title="Test", stage=ProjectStage.DRAFTING,
        )

        for i in range(5):
            store.seo_suggestions.append(SEOBlock(keywords=[f"keyword-{i}"]))

        cm.compress_context(store)

        assert len(store.seo_suggestions) == 1
        assert store.seo_suggestions[0].keywords == ["keyword-4"]
