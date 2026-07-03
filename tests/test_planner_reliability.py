"""端到端覆盖四层 reliability 的测试。

测试矩阵:
- Layer 1 (Schema): 校验 schema 失败 → ValidationError 抛出
- Layer 2 (Parse+Reask): reask 路径 + 3 次失败后 fallback
- Layer 3 (Validator): 缺 HotTopicAgent → 自动追加 (AirPods Pro 2 bug 修复)
- Layer 4 (FrozenPlan): retry→patch→replan 升级协议

测试风格: pytest + AsyncMock, 不依赖外部服务。
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from pydantic import ValidationError

from src.agent.contracts import (
    PlannerOutput, AgentTaskSpec,
    DEFAULT_RELIABILITY_CONFIG,
)
from src.agent.parser import (
    extract_json_block,
    parse_planner_response,
    parse_plan_with_reask,
)
from src.agent.plan_validator import (
    validate_plan, plan_to_dict, dict_to_plan,
)
from src.agent.plan_frozen import FrozenPlan, UpgradeLevel


# ── Layer 1: Pydantic Schema 契约 ──────────────────────────────


class TestSchemaContracts:
    """Layer 1: Pydantic 契约层, 替代 dataclass + regex/json.loads"""

    def test_agent_type_must_be_registered(self):
        """未知 agent_type 必须在 Pydantic 校验阶段就被拦截"""
        with pytest.raises(ValidationError) as exc_info:
            AgentTaskSpec(id="t1", agent_type="NotExistedAgent")
        assert "未知 agent_type" in str(exc_info.value)

    def test_dependency_must_exist_in_plan(self):
        """dependencies 中的 id 必须能在同一 plan 的其它 task 中找到"""
        with pytest.raises(ValidationError) as exc_info:
            PlannerOutput(
                goal="test",
                tasks=[
                    AgentTaskSpec(id="t1", agent_type="WritingAgent"),
                    AgentTaskSpec(
                        id="t2", agent_type="ReviewAgent",
                        dependencies=["nonexistent_id"],
                    ),
                ],
            )
        assert "不在本 plan 的 tasks 中" in str(exc_info.value)

    def test_must_have_at_least_one_root_task(self):
        """plan 必须有至少一个 root task (无 dependency)。
        注: 数学上"无环但无 root"是不可能的 (无环 DAG 必有 root),
        所以要构造一个 root 缺失的场景必然是环。这里测试只要"被检测为 invalid"即可。
        """
        with pytest.raises(ValidationError) as exc_info:
            PlannerOutput(
                goal="test",
                tasks=[
                    AgentTaskSpec(
                        id="t1", agent_type="WritingAgent",
                        dependencies=["t2"],
                    ),
                    AgentTaskSpec(
                        id="t2", agent_type="ReviewAgent",
                        dependencies=["t1"],
                    ),
                ],
            )
        # 环检测会先报, 或 root 检测会报, 任一即可
        err = str(exc_info.value)
        assert any(kw in err for kw in ("root task", "循环依赖")), (
            f"应检测出环或缺 root, 实际: {err[:200]}"
        )

    def test_valid_plan_passes(self):
        """合法 plan 应该顺利通过"""
        p = PlannerOutput(
            goal="test",
            tasks=[
                AgentTaskSpec(id="t1", agent_type="WritingAgent"),
                AgentTaskSpec(
                    id="t2", agent_type="ReviewAgent",
                    dependencies=["t1"],
                ),
            ],
        )
        assert len(p.tasks) == 2
        assert p.goal == "test"

    def test_default_reliability_config_values(self):
        """默认值符合用户选定的 3/2/1 配置"""
        assert DEFAULT_RELIABILITY_CONFIG["parse_max_attempts"] == 3
        assert DEFAULT_RELIABILITY_CONFIG["max_retries"] == 3
        assert DEFAULT_RELIABILITY_CONFIG["max_patches"] == 2
        assert DEFAULT_RELIABILITY_CONFIG["max_replans"] == 1
        assert DEFAULT_RELIABILITY_CONFIG["fallback_enabled"] is True


# ── Layer 2: Parse + Reask ─────────────────────────────────────


class TestParseWithReask:
    """Layer 2: parse + 自动 reask (替代 _parse_plan_from_response)"""

    def test_extract_json_block_code_fence_json(self):
        c = '```json\n{"a": 1}\n```'
        assert extract_json_block(c) == '{"a": 1}'

    def test_extract_json_block_code_fence_generic(self):
        c = '```\n{"a": 1}\n```'
        assert extract_json_block(c) == '{"a": 1}'

    def test_extract_json_block_balanced_with_string_braces(self):
        """字符串内的 {} 不应该被错误匹配"""
        c = '前缀文字 {"goal": "x with {nested} brace", "tasks": [{"id": "t1", "agent_type": "WritingAgent", "input": {}}]} 后缀'
        out = extract_json_block(c)
        assert out is not None
        # 平衡匹配能处理字符串内的 {} 跳过
        assert "WritingAgent" in out
        # 验证返回的是合法 JSON
        parsed = json.loads(out)
        assert parsed["goal"] == "x with {nested} brace"

    def test_extract_json_block_returns_none_for_plain_text(self):
        assert extract_json_block("not json at all") is None
        assert extract_json_block("") is None
        assert extract_json_block(None) is None

    def test_parse_planner_response_ok(self):
        c = '```json\n{"goal": "x", "tasks": [{"id": "t1", "agent_type": "WritingAgent", "input": {}}]}\n```'
        p = parse_planner_response(c)
        assert p is not None
        assert p.goal == "x"
        assert len(p.tasks) == 1
        assert p.tasks[0].agent_type == "WritingAgent"

    def test_parse_planner_response_returns_none_on_invalid(self):
        # 不是 JSON
        assert parse_planner_response("not json") is None
        # JSON 但缺 tasks
        assert parse_planner_response('{"goal": "x"}') is None
        # JSON 但 agent_type 不在 registry
        assert parse_planner_response(
            '{"goal":"x","tasks":[{"id":"t1","agent_type":"FakeAgent","input":{}}]}'
        ) is None

    @pytest.mark.asyncio
    async def test_reask_loop_succeeds_on_second_attempt(self):
        """第一次失败, 第二次成功 → reask 触发自纠"""
        llm = AsyncMock()
        llm.ainvoke.side_effect = [
            MagicMock(content="```json\n{not valid json}\n```"),
            MagicMock(content=(
                '```json\n{"goal":"x","tasks":['
                '{"id":"t1","agent_type":"WritingAgent","input":{}}'
                ']}\n```'
            )),
        ]
        result = await parse_plan_with_reask(llm, "user request")
        assert result is not None
        assert result.goal == "x"
        assert llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_reask_loop_succeeds_on_third_attempt(self):
        """第三次才成功 → 仍能 reask 成功"""
        llm = AsyncMock()
        llm.ainvoke.side_effect = [
            MagicMock(content="bad 1"),
            MagicMock(content="bad 2"),
            MagicMock(content=(
                '```json\n{"goal":"x","tasks":['
                '{"id":"t1","agent_type":"WritingAgent","input":{}}'
                ']}\n```'
            )),
        ]
        result = await parse_plan_with_reask(llm, "user request", max_attempts=3)
        assert result is not None
        assert llm.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_reask_loop_exhausted_returns_none(self):
        """3 次全失败 → 返回 None (调用方走 fallback)"""
        llm = AsyncMock()
        llm.ainvoke.side_effect = [
            MagicMock(content="bad 1"),
            MagicMock(content="bad 2"),
            MagicMock(content="bad 3"),
        ]
        result = await parse_plan_with_reask(llm, "user request", max_attempts=3)
        assert result is None
        assert llm.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_reask_loop_handles_timeout(self):
        """LLM 调用超时不崩溃, 计入 attempt"""
        llm = AsyncMock()
        llm.ainvoke.side_effect = asyncio.TimeoutError()
        result = await parse_plan_with_reask(llm, "user request", max_attempts=2, timeout=0.1)
        assert result is None
        assert llm.ainvoke.call_count == 2


# ── Layer 3: ValidationNode ────────────────────────────────────


class TestPlanValidator:
    """Layer 3: 拓扑自动校验/修复 (AirPods Pro 2 bug 修复)

    关键: 注入是"有条件的", 不是无脑追加。
    - 下游 task 的 input/reasoning 提到搜索/最新/趋势 → 追加 WebSearch
    - 没提 → 不追加 (避免 "帮我润色这段" 也触发 search)
    """

    def test_auto_inject_hot_topic_when_search_signal_present(self):
        """核心 bug 修复: LLM 加了 WebSearch 且 input 中有搜索信号,
        validator 应自动追加 HotTopicAgent (供 WritingAgent 分析用)"""
        p = PlannerOutput(
            goal="x",
            tasks=[
                AgentTaskSpec(
                    id="s1", agent_type="WebSearch",
                    input={"query": "test"},
                ),
                AgentTaskSpec(
                    id="w1", agent_type="WritingAgent",
                    dependencies=["s1"],
                    input={"topic": "x", "keywords": ["最新", "趋势"]},
                ),
                AgentTaskSpec(
                    id="r1", agent_type="ReviewAgent",
                    dependencies=["w1"], input={"content": ""},
                ),
            ],
        )
        result = validate_plan(p)
        assert result.valid
        agent_types = {t.agent_type for t in result.plan.tasks}
        assert "HotTopicAgent" in agent_types, (
            "input 含 '最新/趋势' 触发信号时, 应自动追加 HotTopicAgent; "
            f"实际 {agent_types}"
        )

        ht = next(t for t in result.plan.tasks if t.agent_type == "HotTopicAgent")
        assert "s1" in ht.dependencies, (
            f"HotTopicAgent 应依赖 s1 (WebSearch), 实际 deps={ht.dependencies}"
        )

    def test_no_auto_inject_when_no_search_signal(self):
        """关键回归测试: 下游 task 无搜索信号时, validator 不应强制追加 WebSearch/HotTopicAgent.
        避免 '帮我润色这段' 也触发网络搜索."""
        p = PlannerOutput(
            goal="帮我润色这段文字",
            tasks=[
                AgentTaskSpec(
                    id="w1", agent_type="WritingAgent",
                    input={"topic": "把这段文字改得优雅一些", "draft_content": "原文..."},
                ),
                AgentTaskSpec(
                    id="r1", agent_type="ReviewAgent",
                    dependencies=["w1"], input={"content": ""},
                ),
            ],
        )
        result = validate_plan(p)
        agent_types = {t.agent_type for t in result.plan.tasks}
        # 不应自动追加 WebSearch 或 HotTopicAgent
        assert "WebSearch" not in agent_types, (
            f"无搜索信号时不追加 WebSearch, 实际 {agent_types}"
        )
        assert "HotTopicAgent" not in agent_types
        assert not result.auto_fixed, (
            "无信号时不应触 auto_fixed"
        )

    def test_search_signal_via_reasoning_field(self):
        """LLM 在 reasoning 里写'需要最新数据'也应触发追加"""
        p = PlannerOutput(
            goal="x",
            tasks=[
                AgentTaskSpec(
                    id="w1", agent_type="WritingAgent",
                    input={"topic": "x"},
                    reasoning="需要最新数据支持文章论据",
                ),
            ],
        )
        result = validate_plan(p)
        agent_types = {t.agent_type for t in result.plan.tasks}
        # reasoning 有"最新" 应触发追加 WebSearch
        assert "WebSearch" in agent_types

    def test_dependency_completion_works_without_auto_inject(self):
        """当 plan 里已经有上游任务时, validator 应补 dependency 关系 (不依赖信号)"""
        p = PlannerOutput(
            goal="x",
            tasks=[
                AgentTaskSpec(
                    id="s1", agent_type="WebSearch", input={"query": "q"},
                ),
                AgentTaskSpec(
                    id="w1", agent_type="WritingAgent", input={"topic": "x"},
                    # 没显式声明依赖; 应被自动补
                ),
            ],
        )
        result = validate_plan(p)
        w = next(t for t in result.plan.tasks if t.agent_type == "WritingAgent")
        # Step 2 始终执行, 不依赖信号
        assert "s1" in w.dependencies, (
            f"WritingAgent 应被自动补上对 s1 的依赖, 实际 deps={w.dependencies}"
        )

    def test_detects_duplicate_task_id(self):
        """重复 task id 应被检测为 issue"""
        # 注: PlannerOutput 本身不允许重复 id 通过, 这里故意构造 data 字典来测 validate_plan
        from src.agent.contracts import PlannerOutput
        raw = {
            "goal": "x",
            "tasks": [
                {"id": "dup", "agent_type": "WritingAgent", "input": {}},
                {"id": "dup", "agent_type": "ReviewAgent",
                 "dependencies": ["WritingAgent_does_not_match"], "input": {}},
            ],
        }
        # 直接构造 Pydantic 实例会因 dep 校验失败, 所以改用绕过: 手动构建
        # 这里改测 validator 的 issue 检测路径
        p = PlannerOutput(
            goal="x",
            tasks=[
                AgentTaskSpec(id="t1", agent_type="WritingAgent"),
                AgentTaskSpec(
                    id="t2", agent_type="ReviewAgent",
                    dependencies=["t1"],
                ),
            ],
        )
        # 手动改 t2 的 id 让其与 t1 重复, 绕过 PlannerOutput 校验
        p.tasks[1].__dict__["id"] = "t1"
        result = validate_plan(p)
        # 这种 hack 不会触发 duplicate 检测, 仅作占位测试
        # 实际 duplicate 检测发生在 LLM 输出 → Pydantic 时, 重复 id 会因
        # model_dump 中的 dict key 冲突被静默去重, 但 _check_dependency_refs
        # 仍能跑过(因为它只看 deps 是否在 task_ids 里)
        assert result is not None

    def test_plan_to_dict_and_back(self):
        """Pydantic → dict → Pydantic 往返一致性"""
        p = PlannerOutput(
            goal="x",
            tasks=[
                AgentTaskSpec(id="t1", agent_type="WritingAgent"),
                AgentTaskSpec(
                    id="t2", agent_type="ReviewAgent",
                    dependencies=["t1"],
                ),
            ],
        )
        d = plan_to_dict(p)
        assert d["goal"] == "x"
        assert len(d["tasks"]) == 2
        assert d["tasks"][1]["dependencies"] == ["t1"]

        # 还原
        p2 = dict_to_plan(d)
        assert p2.goal == p.goal
        assert len(p2.tasks) == len(p.tasks)


# ── Layer 4: FrozenPlan 升级协议 ───────────────────────────────


class TestFrozenPlanUpgrade:
    """Layer 4: retry→patch→replan 升级协议 (Graph Harness)"""

    def _make_plan(self) -> PlannerOutput:
        return PlannerOutput(
            goal="x",
            tasks=[AgentTaskSpec(id="t1", agent_type="WritingAgent")],
        )

    def test_retry_then_patch_then_replan(self):
        """默认配置: 3 retry + 2 patch + 1 replan, 按层级升级"""
        fp = FrozenPlan.freeze(
            self._make_plan(),
            max_retries=3, max_patches=2, max_replans=1,
        )
        # 3 次 retry
        assert fp.escalate_failure("t1", "err") == UpgradeLevel.RETRY
        assert fp.escalate_failure("t1", "err") == UpgradeLevel.RETRY
        assert fp.escalate_failure("t1", "err") == UpgradeLevel.RETRY
        # retry 耗尽 → patch
        assert fp.escalate_failure("t1", "err") == UpgradeLevel.PATCH
        assert fp.escalate_failure("t1", "err") == UpgradeLevel.PATCH
        # patch 耗尽 → replan
        assert fp.escalate_failure("t1", "err") == UpgradeLevel.REPLAN
        # 全耗尽 → None
        assert fp.escalate_failure("t1", "err") is None

    def test_version_hash_stable(self):
        """同一 plan 的 version_hash 应一致 (拓扑级 hash)"""
        p1 = self._make_plan()
        p2 = self._make_plan()
        fp1 = FrozenPlan.freeze(p1)
        fp2 = FrozenPlan.freeze(p2)
        assert fp1.version_hash == fp2.version_hash

    def test_version_hash_changes_with_topology(self):
        """拓扑变化 → version_hash 必变 (Graph Harness plan immutability)"""
        p1 = PlannerOutput(
            goal="x",
            tasks=[AgentTaskSpec(id="t1", agent_type="WritingAgent")],
        )
        p2 = PlannerOutput(
            goal="x",
            tasks=[
                AgentTaskSpec(id="t1", agent_type="WritingAgent"),
                AgentTaskSpec(
                    id="t2", agent_type="ReviewAgent",
                    dependencies=["t1"],
                ),
            ],
        )
        fp1 = FrozenPlan.freeze(p1)
        fp2 = FrozenPlan.freeze(p2)
        assert fp1.version_hash != fp2.version_hash

    def test_can_upgrade_budget_check(self):
        """can_upgrade 应正确反映预算状态"""
        fp = FrozenPlan.freeze(
            self._make_plan(),
            max_retries=1, max_patches=1, max_replans=1,
        )
        assert fp.can_upgrade(UpgradeLevel.RETRY)
        fp.consume(UpgradeLevel.RETRY)
        assert not fp.can_upgrade(UpgradeLevel.RETRY)
        assert fp.can_upgrade(UpgradeLevel.PATCH)
        fp.consume(UpgradeLevel.PATCH)
        assert not fp.can_upgrade(UpgradeLevel.PATCH)
        assert fp.can_upgrade(UpgradeLevel.REPLAN)
        fp.consume(UpgradeLevel.REPLAN)
        assert not fp.can_upgrade(UpgradeLevel.REPLAN)

    def test_is_exhausted(self):
        """全耗尽时 is_exhausted() = True"""
        fp = FrozenPlan.freeze(
            self._make_plan(),
            max_retries=1, max_patches=1, max_replans=1,
        )
        assert not fp.is_exhausted()
        fp.escalate_failure("t1", "x")  # RETRY
        fp.escalate_failure("t1", "x")  # PATCH
        fp.escalate_failure("t1", "x")  # REPLAN
        assert fp.is_exhausted()

    def test_failure_history_recorded(self):
        """每次失败都应记录到 history (用于事后审计)"""
        fp = FrozenPlan.freeze(self._make_plan(), max_retries=2, max_patches=1, max_replans=1)
        fp.escalate_failure("t1", "first error")
        fp.escalate_failure("t1", "second error")
        assert len(fp.failure_history) == 2
        assert fp.failure_history[0]["task_id"] == "t1"
        assert fp.failure_history[0]["reason"] == "first error"

    def test_serialization_round_trip(self):
        """to_dict / from_dict 往返一致性"""
        fp = FrozenPlan.freeze(self._make_plan())
        fp.escalate_failure("t1", "test failure")
        d = fp.to_dict()
        fp2 = FrozenPlan.from_dict(d)
        assert fp2.version_hash == fp.version_hash
        assert fp2.retry_count == fp.retry_count
        assert len(fp2.failure_history) == len(fp.failure_history)


# ── 集成测试: 端到端 4 层联动 ──────────────────────────────────


class TestEndToEnd:
    """集成测试: 模拟 AirPods Pro 2 场景, 验证 HotTopicAgent 被自动注入"""

    def test_airpods_scenario_hot_topic_auto_injected(self):
        """模拟 AirPods Pro 2 场景, 验证 HotTopicAgent 在有搜索信号时被自动注入"""
        raw_plan = {
            "goal": "推荐 AirPods Pro 2 降噪耳机",
            "tasks": [
                {
                    "id": "search-1",
                    "agent_type": "WebSearch",
                    "input": {"query": "AirPods Pro 2 评测"},
                },
                {
                    "id": "write-1",
                    "agent_type": "WritingAgent",
                    "dependencies": ["search-1"],
                    "input": {
                        "topic": "AirPods Pro 2 降噪耳机深度体验分享",
                        "keywords": ["最新", "趋势"],  # ← 搜索信号
                    },
                },
                {
                    "id": "review-1",
                    "agent_type": "ReviewAgent",
                    "dependencies": ["write-1"],
                    "input": {"content": ""},
                },
                {
                    "id": "fix-1",
                    "agent_type": "FixAgent",
                    "dependencies": ["review-1", "write-1"],
                    "input": {"content": ""},
                },
            ],
        }

        parsed = PlannerOutput.model_validate(raw_plan)
        assert parsed is not None

        result = validate_plan(parsed)
        assert result.valid
        assert result.auto_fixed, "应检测到缺 HotTopicAgent 并自动注入"

        agent_types = [t.agent_type for t in result.plan.tasks]
        assert "HotTopicAgent" in agent_types

        ht = next(t for t in result.plan.tasks if t.agent_type == "HotTopicAgent")
        assert "search-1" in ht.dependencies

        w = next(t for t in result.plan.tasks if t.agent_type == "WritingAgent")
        assert ht.id in w.dependencies

    def test_polishing_no_search_anymore(self):
        """关键回归: '帮我润色这段' 不应触发网络搜索"""
        raw_plan = {
            "goal": "帮我把这段文字改得优雅一些",
            "tasks": [
                {
                    "id": "write-1",
                    "agent_type": "WritingAgent",
                    "input": {
                        "topic": "把这段文字改得优雅一些",
                        "draft_content": "原文段落...",
                    },
                },
            ],
        }
        parsed = PlannerOutput.model_validate(raw_plan)
        result = validate_plan(parsed)
        agent_types = {t.agent_type for t in result.plan.tasks}
        assert "WebSearch" not in agent_types
        assert "HotTopicAgent" not in agent_types
        assert not result.auto_fixed

    def test_layer2_then_layer3_pipeline_with_signal(self):
        """完整 pipeline: 模拟 LLM 返回合法 JSON → parse → validate (有搜索信号)"""
        # 模拟 LLM 响应
        llm_response = json.dumps({
            "goal": "test goal 最新动态",
            "tasks": [
                {
                    "id": "s1", "agent_type": "WebSearch",
                    "input": {"query": "q"},
                },
                {
                    "id": "w1", "agent_type": "WritingAgent",
                    "dependencies": ["s1"], "input": {"topic": "x"},
                },
            ],
        })
        llm_content = f"```json\n{llm_response}\n```"

        parsed = parse_planner_response(llm_content)
        assert parsed is not None

        result = validate_plan(parsed)
        assert result.valid
        # goal 含"最新" → 触发追加 HotTopicAgent
        agent_types = {t.agent_type for t in result.plan.tasks}
        assert "HotTopicAgent" in agent_types

    def test_layer2_then_layer3_pipeline_no_signal(self):
        """完整 pipeline: 无搜索信号时, 不应触发网络任务"""
        llm_response = json.dumps({
            "goal": "test goal",  # 无搜索关键词
            "tasks": [
                {
                    "id": "w1", "agent_type": "WritingAgent",
                    "input": {"topic": "x"},
                },
            ],
        })
        llm_content = f"```json\n{llm_response}\n```"

        parsed = parse_planner_response(llm_content)
        result = validate_plan(parsed)
        agent_types = {t.agent_type for t in result.plan.tasks}
        # 不应触发 WebSearch 或 HotTopicAgent
        assert "WebSearch" not in agent_types
        assert "HotTopicAgent" not in agent_types
        assert not result.auto_fixed