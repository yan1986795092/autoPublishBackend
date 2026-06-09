"""LangGraph StateGraph 构建器 — 编排全部 Agent 节点

Graph 结构:
  classify_and_decompose
       │ (direct)
       ▼
  schedule_and_execute
       │ (conditional: should_continue_or_finish)
       ├── execute → execute_task_batch
       └── done → merge_and_finish
  execute_task_batch
       │ (conditional: should_continue_or_finish)
       ├── schedule → schedule_and_execute
       └── done → merge_and_finish
  merge_and_finish → END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    classify_and_decompose,
    schedule_and_execute,
    execute_task_batch,
    merge_and_finish,
    should_continue_or_finish,
)
from .state import AgentGraphState


def build_agent_graph() -> StateGraph:
    """构建 Agent 编排 StateGraph"""
    builder = StateGraph(AgentGraphState)

    builder.add_node("classify_and_decompose", classify_and_decompose)
    builder.add_node("schedule_and_execute", schedule_and_execute)
    builder.add_node("execute_task_batch", execute_task_batch)
    builder.add_node("merge_and_finish", merge_and_finish)

    builder.set_entry_point("classify_and_decompose")

    builder.add_edge("classify_and_decompose", "schedule_and_execute")

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
