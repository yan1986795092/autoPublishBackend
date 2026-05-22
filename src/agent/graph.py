"""LangGraph StateGraph 构建器 — 编排全部 Agent 节点

优化后的流程（使用条件边代替显式循环）：
  classify_and_decompose
         │
         ▼
  schedule_and_execute ──(条件边: 有任务?)──→ execute_task_batch ──(条件边: 继续?)──┐
         │                                                               │          │
         │                                                               └──────────┘
         │
         ▼
  merge_and_finish
         │
         ▼
        END
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
    """构建 Agent 编排 StateGraph

    节点说明：
    - classify_and_decompose: 任务分类和分解
    - schedule_and_execute: 调度节点（合并原 find_ready_tasks + check_task_results）
      - 找出可执行的任务
      - 如果有任务，直接触发 execute_task_batch
      - 如果没有任务，进入 merge_and_finish
    - execute_task_batch: 并行执行当前批次任务，执行完后回到 schedule_and_execute
    - merge_and_finish: 合并结果并结束

    条件边：
    - classify_and_decompose → schedule_and_execute (无条件)
    - schedule_and_execute → execute_task_batch (有 ready_tasks)
    - schedule_and_execute → merge_and_finish (无 ready_tasks)
    - execute_task_batch → schedule_and_execute (继续执行)
    - execute_task_batch → merge_and_finish (全部完成)
    """
    builder = StateGraph(AgentGraphState)

    # ── 注册节点 ──
    builder.add_node("classify_and_decompose", classify_and_decompose)
    builder.add_node("schedule_and_execute", schedule_and_execute)
    builder.add_node("execute_task_batch", execute_task_batch)
    builder.add_node("merge_and_finish", merge_and_finish)

    # ── 入口 ──
    builder.set_entry_point("classify_and_decompose")

    # ── 条件边：classify → 是否有任务可执行？ ──
    builder.add_conditional_edges(
        "classify_and_decompose",
        should_continue_or_finish,
        {"execute": "schedule_and_execute", "done": "merge_and_finish"},
    )

    # ── 调度节点的条件边：有任务就执行，否则结束 ──
    builder.add_conditional_edges(
        "schedule_and_execute",
        should_continue_or_finish,
        {"execute": "execute_task_batch", "done": "merge_and_finish"},
    )

    # ── 执行批次后的条件边：继续调度或结束 ──
    builder.add_conditional_edges(
        "execute_task_batch",
        should_continue_or_finish,
        {"execute": "schedule_and_execute", "done": "merge_and_finish"},
    )

    # ── 结束 ──
    builder.add_edge("merge_and_finish", END)

    return builder.compile(
        store=None,
        interrupt_before=[],
        interrupt_after=[],
        debug=False,
    )
