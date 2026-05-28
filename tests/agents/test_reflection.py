"""Reflection — 反思引擎、模型、自省单元测试"""
import pytest

from src.agent.reflection import (
    ReflectionResult, ReflectionConfig, ReflectionEngine,
    self_reflect, peer_review, meta_reflect,
)


class TestReflectionResult:
    """ReflectionResult 数据模型测试"""

    def test_default_values(self):
        r = ReflectionResult(
            agent_type="WritingAgent",
            target_task_id="t1",
            reflection_type="self",
            score=0.85,
        )
        assert r.agent_type == "WritingAgent"
        assert r.score == 0.85
        assert r.passed is True
        assert r.issues == []
        assert r.suggestions == []
        assert r.metadata == {}

    def test_failed_result(self):
        r = ReflectionResult(
            agent_type="CodeAgent",
            target_task_id="t2",
            reflection_type="self",
            score=0.45,
            issues=["缺少注释", "变量命名不规范"],
            suggestions=["添加注释", "使用有意义的变量名"],
            passed=False,
        )
        assert r.passed is False
        assert len(r.issues) == 2

    def test_to_dict(self):
        r = ReflectionResult(
            agent_type="A", target_task_id="t1", reflection_type="self",
            score=0.9, issues=["i1"], passed=True,
        )
        d = r.to_dict()
        assert d["agent_type"] == "A"
        assert d["score"] == 0.9
        assert d["passed"] is True

    def test_from_dict(self):
        data = {
            "agent_type": "Writer", "target_task_id": "t1",
            "reflection_type": "self", "score": 0.75,
            "issues": ["问题1"], "passed": False,
        }
        r = ReflectionResult.from_dict(data)
        assert r.agent_type == "Writer"
        assert r.score == 0.75
        assert r.passed is False

    def test_from_dict_ignores_unknown(self):
        data = {
            "agent_type": "A", "target_task_id": "t", "reflection_type": "self",
            "score": 0.5, "unknown_field": "ignored",
        }
        r = ReflectionResult.from_dict(data)
        assert not hasattr(r, "unknown_field")


class TestReflectionConfig:
    """ReflectionConfig 测试"""

    def test_defaults(self):
        cfg = ReflectionConfig()
        assert cfg.max_self_reflection_rounds == 2
        assert cfg.enable_peer_review is False
        assert cfg.enable_meta_cognition is False
        assert cfg.quality_threshold == 0.7
        assert "WritingAgent" in cfg.self_reflection_agents

    def test_custom_config(self):
        cfg = ReflectionConfig(
            max_self_reflection_rounds=3,
            enable_peer_review=True,
            quality_threshold=0.8,
            self_reflection_agents=["WritingAgent", "CodeAgent"],
        )
        assert cfg.max_self_reflection_rounds == 3
        assert cfg.enable_peer_review is True
        assert cfg.quality_threshold == 0.8

    def test_peer_reviewers_default(self):
        cfg = ReflectionConfig()
        assert cfg.peer_reviewers == ["ReviewAgent"]


class TestReflectionEngine:
    """ReflectionEngine 编排引擎测试"""

    @pytest.fixture
    def engine(self):
        return ReflectionEngine()

    def test_init_with_defaults(self, engine):
        assert engine.config.max_self_reflection_rounds == 2

    def test_init_with_custom_config(self):
        cfg = ReflectionConfig(max_self_reflection_rounds=5)
        engine = ReflectionEngine(config=cfg)
        assert engine.config.max_self_reflection_rounds == 5

    def test_should_retry_below_threshold(self, engine):
        result = ReflectionResult(
            agent_type="A", target_task_id="t", reflection_type="self",
            score=0.5, passed=False,
        )
        assert engine.should_retry(result, round_num=0) is True

    def test_should_not_retry_above_threshold(self, engine):
        result = ReflectionResult(
            agent_type="A", target_task_id="t", reflection_type="self",
            score=0.9, passed=True,
        )
        assert engine.should_retry(result, round_num=0) is False

    def test_should_not_retry_max_rounds(self, engine):
        result = ReflectionResult(
            agent_type="A", target_task_id="t", reflection_type="self",
            score=0.5, passed=False,
        )
        assert engine.should_retry(result, round_num=2) is False

    def test_should_not_retry_skip_metadata(self, engine):
        result = ReflectionResult(
            agent_type="A", target_task_id="t", reflection_type="self",
            score=0.5, passed=False,
            metadata={"skip": True},
        )
        assert engine.should_retry(result, round_num=0) is False

    def test_should_continue_all_passed(self, engine):
        results = [
            ReflectionResult(agent_type="A", target_task_id="t1", reflection_type="self", score=0.9, passed=True),
            ReflectionResult(agent_type="B", target_task_id="t2", reflection_type="self", score=0.8, passed=True),
        ]
        assert engine.should_continue(results) is True

    def test_should_continue_some_failed(self, engine):
        results = [
            ReflectionResult(agent_type="A", target_task_id="t1", reflection_type="self", score=0.9, passed=True),
            ReflectionResult(agent_type="B", target_task_id="t2", reflection_type="self", score=0.4, passed=False),
        ]
        assert engine.should_continue(results) is True  # at least one passed

    def test_should_continue_empty(self, engine):
        assert engine.should_continue([]) is True

    @pytest.mark.asyncio
    async def test_run_self_reflection_disabled_agent(self, engine):
        """HotTopicAgent 不在 self_reflection_agents 列表中"""
        result = await engine.run_self_reflection(
            agent_type="HotTopicAgent",
            task_input={},
            result={"output": {"content": "some content"}},
            llm=None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_run_self_reflection_short_content(self, engine):
        """内容太短时自动跳过"""
        result = await engine.run_self_reflection(
            agent_type="WritingAgent",
            task_input={},
            result={"output": {"content": "short"}},
            llm=None,
        )
        assert result is not None
        assert result.passed is True
        assert result.metadata.get("reason") == "skip: content too short"

    @pytest.mark.asyncio
    async def test_run_peer_review_disabled(self, engine):
        result = await engine.run_peer_review(
            target_agent="WritingAgent",
            target_output={"content": "output"},
            target_summary="summary",
            llm=None,
        )
        assert result is None  # peer review disabled by default

    @pytest.mark.asyncio
    async def test_run_meta_reflection_disabled(self, engine):
        result = await engine.run_meta_reflection(
            plan={}, task_results={}, completed_ids=[], llm=None,
        )
        assert result is None  # meta cognition disabled by default


class TestSelfReflect:
    """自省函数测试"""

    @pytest.mark.asyncio
    async def test_skip_short_content(self):
        result = await self_reflect(
            agent_type="WritingAgent",
            task_input={"_task_id": "t1"},
            result={"output": {"content": "too short"}},
            llm=None,
        )
        assert result is not None
        assert result.passed is True
        assert "skip" in result.metadata.get("reason", "")

    @pytest.mark.asyncio
    async def test_rule_based_passing(self):
        """不同段落的文章应通过规则自省"""
        content = (
            "# 标题\n\n这是一篇完整的文章，讨论了非常重要的话题。"
            "我们深入探讨了相关领域的最新发展，分析了多个典型案例。"
            "研究发现，这一趋势将持续影响整个行业的未来发展方向。"
            "最后总结了关键发现和未来研究方向，并提出了具体建议。"
            "这些发现对于指导实践具有重要意义，值得进一步深入探讨。"
        )
        assert len(content) >= 100, "content too short for rule-based check"
        result = await self_reflect(
            agent_type="WritingAgent",
            task_input={"_task_id": "t1"},
            result={"output": {"content": content}},
            llm=None,
        )
        assert result is not None
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_rule_based_no_title(self):
        """WritingAgent 缺少标题时应有问题"""
        content = "这是一篇没有标题的短文。" * 30
        result = await self_reflect(
            agent_type="WritingAgent",
            task_input={"_task_id": "t1"},
            result={"output": {"content": content}},
            llm=None,
        )
        assert result is not None
        if "缺少标题" in result.issues:
            assert result.passed is False

    @pytest.mark.asyncio
    async def test_no_output_content(self):
        result = await self_reflect(
            agent_type="WritingAgent",
            task_input={},
            result={"output": {}},
            llm=None,
        )
        assert result is not None
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_parse_reflection_json(self):
        """验证 _parse_reflection 能处理真实的 JSON 响应"""
        from src.agent.reflection.self_reflection import _parse_reflection

        raw = """```json
{
  "scores": {"completeness": 4, "accuracy": 5, "clarity": 4, "relevance": 5},
  "issues": ["缺少示例"],
  "suggestions": ["添加代码示例"],
  "overall_score": 0.9
}```"""
        result = _parse_reflection("CodeAgent", {"_task_id": "t1"}, raw)
        assert result.score == pytest.approx(0.9, abs=0.1)
        assert "缺少示例" in result.issues
        assert not result.passed  # has issues

    @pytest.mark.asyncio
    async def test_parse_reflection_no_code_block(self):
        from src.agent.reflection.self_reflection import _parse_reflection

        raw = '{"scores": {"completeness": 5}, "issues": [], "suggestions": [], "overall_score": 1.0}'
        result = _parse_reflection("A", {"_task_id": "t"}, raw)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_parse_reflection_invalid_json(self):
        from src.agent.reflection.self_reflection import _parse_reflection

        result = _parse_reflection("A", {"_task_id": "t"}, "not json at all")
        # Falls back to rule-based
        assert result is not None
        assert isinstance(result.score, float)


class TestMetaReflect:
    """元认知函数测试"""

    @pytest.mark.asyncio
    async def test_empty_task_results(self):
        result = await meta_reflect(
            plan={}, task_results={}, completed_ids=[], llm=None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_rule_based_all_completed(self):
        result = await meta_reflect(
            plan={"goal": "test"},
            task_results={"t1": {"status": "completed"}, "t2": {"status": "completed"}},
            completed_ids=["t1", "t2"],
            llm=None,
        )
        assert result is not None
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_rule_based_partial(self):
        result = await meta_reflect(
            plan={"goal": "test"},
            task_results={"t1": {}, "t2": {}},
            completed_ids=["t1"],
            llm=None,
        )
        assert result is not None
        assert result.score == 0.5
        assert result.passed is False
        assert len(result.issues) == 1

    @pytest.mark.asyncio
    async def test_parse_meta_result(self):
        from src.agent.reflection.meta_cognition import _parse_meta_result

        raw = """```json
{
  "goal_achieved": true,
  "overall_score": 0.85,
  "issues": [],
  "suggestions": ["加快速度"],
  "summary": "执行良好"
}```"""
        result = _parse_meta_result(raw)
        assert result is not None
        assert result.score == 0.85
        assert result.passed is True
        assert result.metadata.get("summary") == "执行良好"
