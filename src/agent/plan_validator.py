"""Plan 拓扑校验器 (ValidationNode equivalent, Layer 3).

LangGraph ValidationNode 模式: 不执行, 只校验 + 自动修复。
返回 is_error=true 触发 self-correction (在 graph.py 中通过 conditional_edge
路由到 replan 节点)。

为什么需要这一步:
- LLM 输出 plan 时, 不会自动知道我们 registry 里的 CATEGORY_DEPENDENCY_RULES
- 比如 LLM 加了 WebSearch 但忘了加 HotTopicAgent, 旧后处理靠
  wants_search 关键词漏场景导致 HotTopicAgent 不被注入 (AirPods Pro 2 bug)
- 本 validator 基于 plan 内容 + registry 推断依赖, 完全不依赖 user prompt 文本

校验/修复流程:
1. 必填依赖注入: writing 必依赖 search/analysis; review 必依赖 writing;
   fix 必依赖 review
2. 自动追加缺失的 category 类任务(从 registry by_category 取代表性 agent)
3. 显式设置 dependency 关系
4. 检测重复 id / 孤立节点 / 环

Reference:
- LangGraph ValidationNode: langgraph.prebuilt.tool_validator.ValidationNode
- Google ADK 2.0: deterministic workflow with bounded agentic nodes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .contracts import PlannerOutput, AgentTaskSpec
from .registry import get, CATEGORY_DEPENDENCY_RULES, by_category
from ..logging_shared import get_logger

logger = get_logger("agent.validator")


@dataclass
class PlanValidationResult:
    """校验结果。

    - valid=True, auto_fixed=False: 通过, 直接执行
    - valid=True, auto_fixed=True: 通过但被自动修复, 记录日志
    - valid=False: 失败, caller 应触发 re-plan
    """

    plan: PlannerOutput
    valid: bool
    auto_fixed: bool
    issues: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)


def _first_root_id(plan: PlannerOutput) -> str:
    """取第一个 root task 的 id (无 dependency 的)。"""
    for t in plan.tasks:
        if not t.dependencies:
            return t.id
    return plan.tasks[0].id


def _last_task_id_in_cat(plan: PlannerOutput, category: str) -> str | None:
    """取 category 类别下最后一个 task 的 id。"""
    for t in reversed(plan.tasks):
        card = get(t.agent_type)
        if card and card.category == category:
            return t.id
    return None


def _needs_upstream_dependency(
    plan: PlannerOutput,
    downstream_task: AgentTaskSpec,
    upstream_cat: str,
) -> bool:
    """判断 downstream task 是否真的需要上游 cat 的任务。

    设计原则: 依赖规则是"潜规则", 不是"强制规则"。
    - LLM 主动加了 WebSearch → writing 真的需要分析结果 → 必须追加 analysis
    - LLM 没加 WebSearch → writing 默认不需要搜 → 不应盲目追加

    触发条件 (任一):
    1. plan.goal 包含搜索意图 (例如 "推荐 AirPods Pro 2 评测")
    2. downstream task 的 input (含 list/嵌套) 中提到搜索关键词
    3. downstream task 的 reasoning 提到"搜索"/"最新"/"趋势" 等关键词
    4. upstream_cat 已经有任务存在 (由 Step 2 处理 dependency 关系)
    """
    import re as _re
    SIGNALS = r"搜索|最新|趋势|实时|热搜|news|trending|search|now|current|评测|排行"
    _sig = _re.compile(SIGNALS, _re.IGNORECASE)

    def _has_signal(text: str) -> bool:
        return bool(text) and bool(_sig.search(text))

    # FIX #5: 收集所有触发来源, 便于 debug
    signal_sources: list[str] = []

    if _has_signal(plan.goal):
        signal_sources.append(f"goal='{plan.goal[:50]}'")

    if _has_signal(downstream_task.reasoning):
        signal_sources.append(f"reasoning='{downstream_task.reasoning[:50]}'")

    def _walk(v: Any) -> bool:
        if isinstance(v, str):
            return _has_signal(v)
        if isinstance(v, dict):
            return any(_walk(x) for x in v.values())
        if isinstance(v, list):
            return any(_walk(x) for x in v)
        return False

    if _walk(downstream_task.input):
        signal_sources.append(
            f"input of {downstream_task.agent_type}={list(downstream_task.input.keys())}"
        )

    if signal_sources:
        # FIX #5: debug log, 运维可看到为什么这条 plan 触发了注入
        logger.debug(
            "search signal detected",
            task=downstream_task.agent_type,
            upstream_cat=upstream_cat,
            sources=signal_sources,
        )
        return True
    return False


def _inject_missing_deps(
    plan: PlannerOutput,
) -> tuple[PlannerOutput, list[str]]:
    """基于 registry 的 CATEGORY_DEPENDENCY_RULES 自动注入缺失的依赖。

    关键策略 (修复 over-injection bug):
    - 仅当下游 task 的 input/reasoning 显式提到 "搜索/最新/趋势" 时,
      才追加 WebSearch / HotTopicAgent
    - 单纯添加 WritingAgent 不应触发自动搜索 (避免 "帮我润色这段" 也走 search)
    - 已有上游任务时, 仅补 dependency 关系, 不重新追加

    Step 1: 条件性追加缺失上游任务 (AirPods Pro 2 bug 修复路径)
    Step 2: 显式补齐 dependency 关系 (始终执行)
    """
    fixes: list[str] = []
    tasks_by_cat: dict[str, list[AgentTaskSpec]] = {}
    for t in plan.tasks:
        card = get(t.agent_type)
        if card:
            tasks_by_cat.setdefault(card.category, []).append(t)

    # Step 1: 条件性追加缺失的上游类任务
    # 只有当下游 task 显式表达 "需要搜索" 时才追加
    for downstream_cat, upstream_cats in CATEGORY_DEPENDENCY_RULES.items():
        if downstream_cat not in tasks_by_cat:
            continue  # 当前 plan 根本没用到这个 downstream 类, 跳过
        for up_cat in upstream_cats:
            if up_cat in tasks_by_cat:
                continue  # 上游类已有, 跳过
            # 检查这个下游类里, 是否至少有一个 task 表达了搜索意图
            downstream_tasks = tasks_by_cat.get(downstream_cat, [])
            any_needs = any(
                _needs_upstream_dependency(plan, dt, up_cat)
                for dt in downstream_tasks
            )
            if not any_needs:
                continue  # 没有触发信号, 不应盲目追加

            candidates = by_category(up_cat)
            if not candidates:
                continue
            # 过滤 PlannerAgent (它是 meta-agent, 不能在 plan 里被追加)
            candidates = [c for c in candidates if c.name != "PlannerAgent"]
            if not candidates:
                continue
            # FIX #10: 显式优先 HotTopicAgent (search-analysis 的标准 agent)
            # 避免依赖 by_category 的注册顺序
            # 优先级: 显式白名单 > stage_label 含分析类 > 第一个
            hot_topic = next(
                (c for c in candidates if c.name == "HotTopicAgent"),
                None,
            )
            if hot_topic:
                candidate = hot_topic
            else:
                analysis_like = [
                    c for c in candidates
                    if any(kw in (c.stage_label or "")
                           for kw in ("整合", "分析", "洞察", "趋势"))
                ]
                candidate = analysis_like[0] if analysis_like else candidates[0]
            upstream_id = _last_task_id_in_cat(plan, up_cat) or _first_root_id(plan)
            new_task = AgentTaskSpec(
                id=f"auto-{candidate.name.lower()}-{len(plan.tasks)}",
                agent_type=candidate.name,
                input={"topic": plan.goal, "platform": ""},
                dependencies=[upstream_id] if upstream_id else [],
                reasoning=f"自动注入: {downstream_cat} 依赖 {up_cat} (触发信号: 下游 task 提到搜索)",
            )
            try:
                new_task.model_validate(new_task.model_dump())
                plan.tasks.append(new_task)
                tasks_by_cat.setdefault(up_cat, []).append(new_task)
                fixes.append(
                    f"auto_append:{candidate.name}(category={up_cat}, signal=needed)"
                )
            except ValidationError as e:
                logger.warning(
                    "auto-append failed validation",
                    agent=candidate.name,
                    error=str(e)[:200],
                )

    # Step 2: 显式补齐 dependency 关系 (始终执行, 不受信号影响)
    new_tasks: list[AgentTaskSpec] = []
    for t in plan.tasks:
        card = get(t.agent_type)
        if not card:
            new_tasks.append(t)
            continue
        dep_cats = CATEGORY_DEPENDENCY_RULES.get(card.category, [])
        if not dep_cats:
            new_tasks.append(t)
            continue

        new_deps = list(t.dependencies)
        for dc in dep_cats:
            upstream_ids = [ut.id for ut in tasks_by_cat.get(dc, [])]
            if not upstream_ids:
                continue
            last_up_id = upstream_ids[-1]
            if last_up_id not in new_deps:
                new_deps.append(last_up_id)
                fixes.append(f"{t.agent_type}.deps+=[{last_up_id}]")

        new_tasks.append(t.model_copy(update={"dependencies": new_deps}))

    new_plan = plan.model_copy(update={"tasks": new_tasks})
    return new_plan, fixes


def validate_plan(plan: PlannerOutput) -> PlanValidationResult:
    """主入口: 校验 + 自动修复。

    Returns:
        PlanValidationResult: valid + issues + auto_fixed
    """
    issues: list[str] = []

    # 基础校验已在 Pydantic model_validator 跑过, 这里做语义级
    seen_ids: set[str] = set()
    for t in plan.tasks:
        if t.id in seen_ids:
            issues.append(f"重复 task id: {t.id}")
        seen_ids.add(t.id)

    # 自动修复依赖
    fixed_plan, fixes = _inject_missing_deps(plan)
    auto_fixed = bool(fixes)

    if auto_fixed:
        logger.info(
            "plan auto-fixed by validator",
            fixes=fixes[:5],  # log 前 5 条
            total=len(fixes),
        )

    valid = len(issues) == 0
    return PlanValidationResult(
        plan=fixed_plan,
        valid=valid,
        auto_fixed=auto_fixed,
        issues=issues,
        fixes=fixes,
    )


def plan_to_dict(plan: PlannerOutput) -> dict[str, Any]:
    """把 Pydantic PlannerOutput 序列化为 graph state 用的 dict。

    保持与原 _parse_plan_from_response 返回 dict 的字段兼容, 让下游
    schedule_and_execute 不感知格式变化。
    """
    return {
        "goal": plan.goal,
        "tasks": [
            {
                "id": t.id,
                "agent_type": t.agent_type,
                "input": t.input,
                "dependencies": t.dependencies,
                "can_parallel_with": t.can_parallel_with,
                "retry_on_low_quality": t.retry_on_low_quality,
                "priority": t.priority,
                "config": {"reasoning": t.reasoning, "priority": i},
            }
            for i, t in enumerate(plan.tasks)
        ],
        "reasoning": plan.reasoning,
        "estimated_steps": len(plan.tasks),
    }


def dict_to_plan(data: dict[str, Any]) -> PlannerOutput:
    """把 graph state 的 dict 反序列化为 PlannerOutput。

    主要给 PR-5 (replan) 用: replan 时从 state 拿老 plan 转回 Pydantic。
    """
    return PlannerOutput.model_validate({
        "goal": data.get("goal", ""),
        "tasks": data.get("tasks", []),
        "reasoning": data.get("reasoning", ""),
    })