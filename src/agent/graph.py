"""LangGraph StateGraph 构建器 — 编排全部 Agent 节点

Graph 结构 (PR-6 增加 validate_plan / replan_plan):
  understand_intent
       │ (direct)
       ▼
  plan_tasks                     ← 内部调用 parse_plan_with_reask（已完成"解析+拓扑校验+LLM自纠"）
       │ (direct)
       ▼
  validate_plan_node             ← Layer 3 ValidationNode 入口 (拓扑二次校验)
       │ (conditional: plan_valid?)
       ├── valid   → schedule_and_execute
       └── invalid → replan_plan
                        │ (conditional: replan 成功?)
                        ├── success → schedule_and_execute
                        └── exhausted → merge_and_finish (走现有 plan)
  schedule_and_execute
       │ (conditional: should_continue_or_finish)
       ├── execute → execute_task_batch
       └── done → merge_and_finish
  execute_task_batch
       │ (conditional: should_continue_or_finish)
       ├── schedule → schedule_and_execute
       └── done → merge_and_finish
  merge_and_finish → END

Reference:
- LangGraph ValidationNode pattern
- Graph Harness 三承诺 (arxiv 2604.11378)
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    understand_intent,
    plan_tasks,
    validate_plan_node,
    replan_plan_node,
    schedule_and_execute,
    execute_task_batch,
    merge_and_finish,
    should_continue_or_finish,
)
from .state import AgentGraphState


def _after_validate(state: AgentGraphState) -> str:
    """validate_plan_node 后的路由。

    - plan_valid=True → 进入执行
    - plan_valid=False → 触发 replan
    """
    plan_valid = state.get("plan_valid")
    logger.info(
        "_after_validate: plan_valid=%s (type=%s)",
        plan_valid, type(plan_valid).__name__,
    )
    if plan_valid:
        return "execute"
    return "replan"


def _after_replan(state: AgentGraphState) -> str:
    """replan_plan_node 后的路由。

    - 重新拿到合法 plan → 进入执行
    - 跳过(预算耗尽) → 也进入执行(放行现有 plan)
    - plan 还是 None(罕见) → 直接 merge_and_finish
    """
    if state.get("plan_valid") is False and not state.get("plan"):
        return "finish"
    return "execute"


@lru_cache(maxsize=1)
def build_agent_graph() -> CompiledStateGraph:
    """构建 Agent 编排 StateGraph(缓存编译结果, 避免重复构建)"""
    builder = StateGraph(AgentGraphState)

    builder.add_node("understand_intent", understand_intent)
    builder.add_node("plan_tasks", plan_tasks)
    builder.add_node("validate_plan", validate_plan_node)
    builder.add_node("replan_plan", replan_plan_node)
    builder.add_node("schedule_and_execute", schedule_and_execute)
    builder.add_node("execute_task_batch", execute_task_batch)
    builder.add_node("merge_and_finish", merge_and_finish)

    builder.set_entry_point("understand_intent")

    builder.add_edge("understand_intent", "plan_tasks")
    builder.add_edge("plan_tasks", "validate_plan")

    builder.add_conditional_edges(
        "validate_plan",
        _after_validate,
        {"execute": "schedule_and_execute", "replan": "replan_plan"},
    )

    builder.add_conditional_edges(
        "replan_plan",
        _after_replan,
        {
            "execute": "schedule_and_execute",
            "finish": "merge_and_finish",
        },
    )

    builder.add_conditional_edges(
        "schedule_and_execute",
        should_continue_or_finish,
        {"execute": "execute_task_batch", "done": "merge_and_finish"},
    )

    builder.add_conditional_edges(
        "execute_task_batch",
        should_continue_or_finish,
        {"execute": "schedule_and_execute", "done": "merge_and_finish"},
    )

    builder.add_edge("merge_and_finish", END)

    return builder.compile(
        store=None,
        interrupt_before=[],
        interrupt_after=[],
        debug=False,
    )