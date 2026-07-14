"""AgentGraphState definitions for the LangGraph workflow."""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class AgentGraphState(TypedDict):
    # Input
    user_request: str
    context: dict

    # Skill context
    skill_system_prompt: str
    skill_name: str

    # Planning
    plan: Optional[dict]
    plan_reasoning: str

    # Execution state
    task_queue: list[dict]
    completed_task_ids: list[str]
    failed_task_ids: list[str]
    task_results: dict

    # Current batch
    ready_tasks: list[dict]
    concurrency_limit: int

    # Observability
    thinking_steps: list[dict]
    trace_task_id: str
    start_time: float

    # Final output
    final_output: Optional[dict]
    error: Optional[str]
    status: str

    # Intent understanding
    understanding: Optional[dict]
    understanding_raw: Optional[str]

    # Runtime-only references
    _async_task_id: str
    _task_queue_ref: Any
    _llm_models: dict
    _mcp_tools: list
    _streaming_mode: bool

    # Plan validation
    plan_valid: bool

    # Review / fix loop control
    _review_fix_cycles: int
    _loop_created_in_cycle: bool
    _last_review_passed: bool

    # Shared batch results
    _shared_results: dict

    # Latest review feedback for FixAgent
    _review_feedback_for_fix: str

    # Preview sync data (title + tags + body, from preview_sync_node)
    _preview_data: dict
