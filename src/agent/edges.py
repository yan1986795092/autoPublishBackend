"""条件边路由函数 — 集中管理所有 conditional edges

注意：基础路由函数 should_continue_or_finish 定义在 nodes.py 中，
此处只包含扩展路由（反思、HITL 等）。
"""
from __future__ import annotations

import logging

from .nodes import should_continue_or_finish
from .state import AgentGraphState

logger = logging.getLogger(__name__)


def should_reflect_or_continue(state: AgentGraphState) -> str:
    """Phase 3: 判断是否需要进入反思回路

    返回值：
    - "reflect": 需要对刚完成的批次进行反思
    - "schedule": 继续调度执行
    - "done": 全部完成
    """
    if not state.get("enable_reflection"):
        return should_continue_or_finish(state)

    reflection_results = state.get("reflection_results", [])
    recent_results = state.get("task_results", {})

    if not recent_results:
        return should_continue_or_finish(state)

    # 有任务结果但还没有反思 → 进入反思
    if not reflection_results:
        return "reflect"

    # 检查是否有未通过自省且还有重试次数的任务
    reflection_rounds = state.get("self_reflection_rounds", {})
    for r in reflection_results:
        if r.get("reflection_type") == "self" and not r.get("passed", True):
            task_id = r.get("target_task_id", "")
            current_round = reflection_rounds.get(task_id, 0)
            if current_round < 2:
                return "reflect"

    return should_continue_or_finish(state)


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
                return "execute"

    return "done"
