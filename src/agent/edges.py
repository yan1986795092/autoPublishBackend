"""条件边路由函数 — 集中管理所有 conditional edges

从 nodes.py 中提取，使路由逻辑可集中维护和测试。
"""
from __future__ import annotations

import logging

from .state import AgentGraphState

logger = logging.getLogger(__name__)


def should_continue_or_finish(state: AgentGraphState) -> str:
    """条件边路由函数：判断是否继续执行任务

    返回值：
    - "execute": 有可执行的任务，继续执行
    - "done": 所有任务已完成，或无可执行任务且无死锁

    逻辑：
    1. 如果 ready_tasks 有任务 → "execute"
    2. 如果没有 ready_tasks 且还有未完成任务 → "done"（死锁或依赖阻塞）
    3. 所有任务都已完成或失败 → "done"
    """
    task_queue = state.get("task_queue", [])
    completed = len(state.get("completed_task_ids", []))
    failed = len(state.get("failed_task_ids", []))
    ready = state.get("ready_tasks", [])
    total_tasks = len(task_queue)
    total_done = completed + failed

    # 有可执行的任务
    if ready:
        return "execute"

    # 所有任务已完成或失败
    if total_done >= total_tasks:
        return "done"

    # 死锁检测
    if total_done < total_tasks:
        logger.warning(
            "No ready tasks (%d total, %d done/failed). Forcing exit.",
            total_tasks, total_done,
        )
        return "done"

    return "done"


def should_reflect_or_continue(state: AgentGraphState) -> str:
    """Phase 3: 判断是否需要进入反思回路

    返回值：
    - "reflect": 需要对刚完成的批次进行反思
    - "schedule": 继续调度执行
    - "done": 全部完成
    """
    if not state.get("enable_reflection"):
        # 未启用反思，直接使用原始路由逻辑
        return should_continue_or_finish(state)

    # 检查是否有刚完成的、需要反思的任务
    reflection_results = state.get("reflection_results", [])
    recent_results = state.get("task_results", {})

    # 只在有完成的任务且未超过总任务数时继续
    if not recent_results:
        return should_continue_or_finish(state)

    # 如果有任务结果但还没有反思，进入反思
    if not reflection_results:
        return "reflect"

    # 检查最近的反思结果是否有需要重试的
    reflection_rounds = state.get("self_reflection_rounds", {})
    for r in reflection_results:
        task_id = r.get("target_task_id", "")
        if r.get("reflection_type") == "self" and not r.get("passed", True):
            current_round = reflection_rounds.get(task_id, 0)
            if current_round < 2:  # max_self_reflection_rounds
                return "reflect"

    return should_continue_or_finish(state)


def should_check_hitl_gates(state: AgentGraphState) -> str:
    """Phase 4: 检查是否需要等待人类决策

    返回值：
    - "execute": 可以继续执行
    - "wait": 需要等待人类决策
    """
    if not state.get("enable_hitl"):
        return "execute"

    pending_gates = state.get("pending_gates", [])
    if pending_gates:
        return "wait"

    return "execute"


def should_retry_after_reflection(state: AgentGraphState) -> str:
    """反思完成后判断是否需要重试任务

    返回值：
    - "execute": 有任务需要重试（回到 schedule_and_execute）
    - "done": 无需重试（进入 merge_and_finish）
    """
    reflection_results = state.get("reflection_results", [])
    reflection_rounds = state.get("self_reflection_rounds", {})

    for r in reflection_results:
        if r.get("reflection_type") == "self" and not r.get("passed", True):
            task_id = r.get("target_task_id", "")
            current_round = reflection_rounds.get(task_id, 0)
            if current_round < 2:
                return "execute"  # 需要重试

    return "done"
