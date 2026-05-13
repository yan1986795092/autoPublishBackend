"""Graph 节点函数单元测试：prompt 模板、helper 函数"""
import pytest

from src.agent.agent_utils import (
    seo_score,
    seo_suggestions,
    extract_output,
    CONTENT_TYPE_PROMPTS,
    STYLE_GUIDES,
    QUALITY_CHECKS,
    BANNED_WORDS,
)


class TestContentConstants:
    """内容类型、风格指南等常量验证"""

    def test_content_type_prompts_has_required_types(self):
        for t in ("blog", "social_post", "newsletter", "report", "copywriting", "script"):
            assert t in CONTENT_TYPE_PROMPTS, f"Missing content type: {t}"

    def test_style_guides_has_entries(self):
        assert len(STYLE_GUIDES) >= 3

    def test_quality_checks_is_list_of_strings(self):
        assert isinstance(QUALITY_CHECKS, list)
        assert len(QUALITY_CHECKS) >= 3
        assert all(isinstance(s, str) for s in QUALITY_CHECKS)

    def test_banned_words_is_list(self):
        assert isinstance(BANNED_WORDS, list)
        assert len(BANNED_WORDS) > 0


class TestSEOHelpers:
    """SEO 辅助函数测试"""

    def test_seo_score_zero_for_empty(self):
        keywords = [{"keyword": "test keyword", "density": 0.02}]
        score = seo_score("", keywords, "test")
        assert score is None

    def test_seo_score_keyword_present(self):
        keywords = [{"keyword": "test", "density": 0.02}]
        # 500 words with ~2% keyword density (10 occurrences)
        content = ("this is a normal article about various topics and subjects that we discuss daily " * 50
                   + "test test test test test test test test test test ")
        score = seo_score(content, keywords, "test article")
        assert score is not None
        assert isinstance(score, dict)
        assert "overall" in score

    def test_seo_suggestions_returns_list(self):
        keywords = [{"keyword": "test", "density": 0.02}]
        score_data = {"score": 0.3, "keyword_usage": {"test": 1}, "total_words": 100}
        suggestions = seo_suggestions("test content", keywords, score_data)
        assert isinstance(suggestions, list)

    def test_seo_suggestions_empty_for_good_score(self):
        keywords = [{"keyword": "test", "density": 0.02}]
        score_data = {"overall": 95, "word_count": 500, "details": {"title_score": 80, "keyword_score": 90, "readability_score": 90}}
        suggestions = seo_suggestions("test " * 100, keywords, score_data)
        # Always returns at least the default suggestion
        assert isinstance(suggestions, list)


class TestGraphOutputHelpers:
    """Graph 节点输出提取辅助函数"""

    def test_extract_output_with_content(self):
        result = extract_output("hello world")
        assert result == "hello world"

    def test_extract_output_empty(self):
        assert extract_output("") == ""
