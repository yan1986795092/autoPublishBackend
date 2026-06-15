"""AgentGraphState — LangGraph StateGraph 的状态定义"""
from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict


class AgentGraphState(TypedDict):
    # === 输入 ===
    user_request: str
    context: dict

    # === Skill 上下文 ===
    skill_system_prompt: str       # Skill 的 system prompt 指令
    skill_name: str               # Skill 名称

    # === 计划 ===
    plan: Optional[dict]          # ExecutionPlan 的 dict 形式
    plan_reasoning: str

    # === 执行状态 ===
    task_queue: list[dict]        # 待执行的子任务列表（有序）
    completed_task_ids: list[str] # 已完成的任务 ID
    failed_task_ids: list[str]    # 失败的任务 ID
    task_results: dict            # task_id -> AgentResult dict

    # === 当前批次 ===
    ready_tasks: list[dict]       # 依赖已就绪、可执行的任务
    concurrency_limit: int        # 最大并行数

    # === 可观测性 ===
    thinking_steps: list[dict]
    trace_task_id: str
    start_time: float

    # === 输出 ===
    final_output: Optional[dict]
    error: Optional[str]
    status: str                   # running / completed / failed / partial

    # === 意图理解（LLM 分析结果，替代旧的 intent_analysis） ===
    understanding: Optional[dict]    # understand_intent() 的结构化输出
    understanding_raw: Optional[str] # LLM 原始响应（调试用）

    # === 系统引用（不序列化） ===
    _async_task_id: str
    _task_queue_ref: Any          # TaskQueue 实例引用
    _llm_models: dict             # agent_type -> ChatOpenAI
    _mcp_tools: list              # MCP Server 加载的工具（LangChain BaseTool 列表）
    _streaming_mode: bool
