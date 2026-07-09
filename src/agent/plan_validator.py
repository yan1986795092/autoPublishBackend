"""Plan 拓扑校验器 (Layer 3)。

纯报告模式：只校验并报告问题，不自动修复。
问题由 parse_plan_with_reask 注入下轮 LLM 重新生成。

校验项：
1. 重复 task id
2. 重复 agent_type
3. WebSearch 无 SearchAnalyzer（需配合）
4. ReviewAgent 无 FixAgent（需配合）
5. 孤立依赖（dep 指向不存在的 task）
6. 自引用（task 依赖自身）
7. 循环依赖（DFS 环检测）
8. 无 root task（全有依赖）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import PlannerOutput


@dataclass
class PlanValidationResult:
    """校验结果。

    - valid=True: plan 通过所有校验
    - valid=False: plan 有 issues，caller 将 issues 注入 LLM 重新生成
    """

    plan: PlannerOutput
    valid: bool
    issues: list[str] = field(default_factory=list)


def validate_plan(plan: PlannerOutput) -> PlanValidationResult:
    """校验 plan，返回所有发现的问题列表（不自动修复）。

    所有问题都应反馈给 LLM，让其在下一轮重新生成。
    """
    issues: list[str] = []
    task_ids: set[str] = {t.id for t in plan.tasks}

    # 1. 重复 id
    seen_ids: set[str] = set()
    for t in plan.tasks:
        if t.id in seen_ids:
            issues.append(f"重复 task id: '{t.id}'")
        seen_ids.add(t.id)

    # 2. 重复 agent_type
    seen_types: set[str] = set()
    for t in plan.tasks:
        if t.agent_type in seen_types:
            issues.append(f"重复 agent_type: '{t.agent_type}'（id='{t.id}'）")
        seen_types.add(t.agent_type)

    # 3. WebSearch 无 SearchAnalyzer
    has_websearch = any(t.agent_type == "WebSearch" for t in plan.tasks)
    has_analyzer = any(t.agent_type == "SearchAnalyzer" for t in plan.tasks)
    if has_websearch and not has_analyzer:
        issues.append(
            "WebSearch 缺少 SearchAnalyzer。"
            "搜索结果必须经过 SearchAnalyzer 分析才能给 WritingAgent 使用。"
            "请添加 SearchAnalyzer 并设置 dependencies=[WebSearch的id]。"
        )

    # 4. ReviewAgent 无 FixAgent
    has_writing = any(t.agent_type == "WritingAgent" for t in plan.tasks)
    has_review = any(t.agent_type == "ReviewAgent" for t in plan.tasks)
    has_fix = any(t.agent_type == "FixAgent" for t in plan.tasks)
    if has_writing and has_review and not has_fix:
        issues.append(
            "ReviewAgent 缺少 FixAgent。"
            "审核完成后必须跟 FixAgent 根据审核意见修复内容。"
            "请添加 FixAgent 并设置 dependencies=[ReviewAgent的id]。"
        )

    # 5. 孤立依赖
    for t in plan.tasks:
        for d in t.dependencies:
            if d not in task_ids:
                issues.append(f"task '{t.id}' 的依赖 '{d}' 不存在（应引用本 plan 中其他 task 的 id）")

    # 6. 自引用
    for t in plan.tasks:
        if t.id in t.dependencies:
            issues.append(f"task '{t.id}' 依赖自身，请移除 dependencies 中的 '{t.id}'")

    # 7. 循环依赖（DFS 环检测）
    adj: dict[str, list[str]] = {t.id: list(t.dependencies) for t in plan.tasks}
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {tid: WHITE for tid in task_ids}

    def _dfs(node: str, path: list[str]) -> None:
        if color[node] == GRAY:
            cycle = " -> ".join(path[path.index(node):] + [node])
            issues.append(f"存在循环依赖: {cycle}")
            return
        if color[node] == BLACK:
            return
        color[node] = GRAY
        for nxt in adj.get(node, []):
            if nxt in color:
                _dfs(nxt, path + [nxt])
        color[node] = BLACK

    for tid in task_ids:
        if color[tid] == WHITE:
            _dfs(tid, [tid])

    # 8. 无 root task
    roots = [t for t in plan.tasks if not t.dependencies]
    if not roots and plan.tasks:
        issues.append("所有 task 都有 dependencies，plan 必须至少有一个根任务（dependencies=[]）")

    valid = len(issues) == 0
    return PlanValidationResult(plan=plan, valid=valid, issues=issues)


def plan_to_dict(plan: PlannerOutput) -> dict[str, Any]:
    """把 Pydantic PlannerOutput 序列化为 graph state 用的 dict。"""
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
    """把 graph state 的 dict 反序列化为 PlannerOutput。"""
    return PlannerOutput.model_validate({
        "goal": data.get("goal", ""),
        "tasks": data.get("tasks", []),
        "reasoning": data.get("reasoning", ""),
    })
