"""LangGraph StateGraph 构建器 — 编排全部 Agent 节点

混合架构：
  - 中心调度：DAG 依赖驱动，schedule_and_execute → execute_task_batch
  - 点对点：通过 MessageBus 支持（Phase 2，通过 task_input 注入 handler）
  - 反思回路：execute → reflect → retry（Phase 3，条件启用）
  - 人机协同：通过 HITL gates + TaskQueue（Phase 4，handler 层处理）

最终 Graph 结构：
  classify_and_decompose
       │ (direct)
       ▼
  schedule_and_execute
       │ (conditional: should_continue_or_finish)
       ├── execute → execute_task_batch
       └── done → merge_and_finish
  execute_task_batch
       │ (conditional: should_reflect_or_continue)
       ├── reflect → reflect_on_task_batch
       ├── schedule → schedule_and_execute
       └── done → merge_and_finish
  reflect_on_task_batch
       │ (conditional: should_retry_after_reflection)
       ├── execute → schedule_and_execute (retry failed tasks)
       └── done → merge_and_finish
  merge_and_finish → END

  （check_hitl_gates 节点注册但不在热路径中，为未来 Graph 级 HITL 预留）
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    classify_and_decompose,
    schedule_and_execute,
    execute_task_batch,
    merge_and_finish,
    reflect_on_task_batch,
    check_hitl_gates,
    should_continue_or_finish,
)
from .edges import (
    should_reflect_or_continue,
    should_retry_after_reflection,
)
from .state import AgentGraphState


def build_agent_graph() -> StateGraph:
    """构建 Agent 编排 StateGraph

    所有 Phase 2-4 特性均为条件启用：
    - use_message_bus=False → 无 P2P 通信
    - enable_reflection=False → reflect_on_task_batch 跳过
    - enable_hitl=False → check_hitl_gates 纯透传
    """
    builder = StateGraph(AgentGraphState)

    # ── 注册节点 ──
    builder.add_node("classify_and_decompose", classify_and_decompose)
    builder.add_node("schedule_and_execute", schedule_and_execute)
    builder.add_node("execute_task_batch", execute_task_batch)
    builder.add_node("reflect_on_task_batch", reflect_on_task_batch)       # Phase 3
    builder.add_node("check_hitl_gates", check_hitl_gates)                 # Phase 4（观测节点）
    builder.add_node("merge_and_finish", merge_and_finish)

    # ── 入口 ──
    builder.set_entry_point("classify_and_decompose")

    # ── classify → schedule（直接边）──
    builder.add_edge("classify_and_decompose", "schedule_and_execute")

    # ── schedule → execute 或 done ──
    builder.add_conditional_edges(
        "schedule_and_execute",
        should_continue_or_finish,
        {"execute": "execute_task_batch", "done": "merge_and_finish"},
    )

    # ── execute_task_batch → reflect / schedule / done ──
    builder.add_conditional_edges(
        "execute_task_batch",
        should_reflect_or_continue,
        {
            "reflect": "reflect_on_task_batch",
            "schedule": "schedule_and_execute",
            "done": "merge_and_finish",
        },
    )

    # ── reflect_on_task_batch → schedule（重试）或 done ──
    builder.add_conditional_edges(
        "reflect_on_task_batch",
        should_retry_after_reflection,
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
