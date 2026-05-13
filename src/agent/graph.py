"""LangGraph StateGraph 构建器 — 编排全部 Agent 节点"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    check_task_results,
    classify_and_decompose,
    execute_task_batch,
    find_ready_tasks,
    merge_and_finish,
    should_continue,
)
from .state import AgentGraphState


def build_agent_graph() -> StateGraph:
    """构建 Agent 编排 StateGraph

    流程:
      classify_and_decompose → ──→ find_ready_tasks → execute_task_batch
                                 ↑                      │
                                 │                      ↓
                                 └── should_continue ← check_task_results
                                              │
                                              ↓
                                        merge_and_finish → END
    """
    builder = StateGraph(AgentGraphState)

    # ── 注册节点 ──
    builder.add_node("classify_and_decompose", classify_and_decompose)
    builder.add_node("find_ready_tasks", find_ready_tasks)
    builder.add_node("execute_task_batch", execute_task_batch)
    builder.add_node("check_task_results", check_task_results)
    builder.add_node("merge_and_finish", merge_and_finish)

    # ── 入口 ──
    builder.set_entry_point("classify_and_decompose")

    # ── 条件边：classify → 是否有任务可执行？ ──
    builder.add_conditional_edges(
        "classify_and_decompose",
        should_continue,
        {"continue": "find_ready_tasks", "done": "merge_and_finish"},
    )

    # ── 核心循环：find → execute → check → should_continue ──
    builder.add_edge("find_ready_tasks", "execute_task_batch")
    builder.add_edge("execute_task_batch", "check_task_results")

    builder.add_conditional_edges(
        "check_task_results",
        should_continue,
        {"continue": "find_ready_tasks", "done": "merge_and_finish"},
    )

    # ── 结束 ──
    builder.add_edge("merge_and_finish", END)

    return builder.compile()
