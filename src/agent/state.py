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

    # === 可观测性（保持与原有 ThinkingTrace 一致的结构）===
    thinking_steps: list[dict]
    trace_task_id: str
    start_time: float

    # === 人机协作 ===
    pending_decision: Optional[dict]
    decision_result: Optional[str]

    # === 输出 ===
    final_output: Optional[dict]
    error: Optional[str]
    status: str                   # running / completed / failed / partial

    # === 系统引用（不序列化） ===
    _async_task_id: str
    _task_queue_ref: Any          # TaskQueue 实例引用
    _llm_models: dict             # agent_type -> ChatOpenAI
    _mcp_tools: list              # MCP Server 加载的工具（LangChain BaseTool 列表）
    _streaming_mode: bool = False  # 是否开启实时流式推送（LLM token 级）
    _agent_registry: Any = None    # AgentRegistry 实例（Phase 1）
    _tool_registry: Any = None     # ToolRegistry 实例（Tool System）

    # === P2P 通信（Phase 2） ===
    use_message_bus: bool = False
    _message_bus: Any = None       # MessageBus 实例
    _messages: list[dict] = []     # 消息日志（可序列化）

    # === 反思（Phase 3） ===
    enable_reflection: bool = False
    reflection_config: Any = None  # ReflectionConfig dict
    _reflection_engine: Any = None # ReflectionEngine 实例
    reflection_results: list[dict] = []   # serialized ReflectionResult
    self_reflection_rounds: dict = {}     # task_id -> round count

    # === HITL（Phase 4） ===
    enable_hitl: bool = False
    _hitl_manager: Any = None      # HITLManager 实例
    pending_gates: list[dict] = [] # 序列化的 pending gates

    # === Context Management (Layered Multi-Agent) ===
    _context_store: Any = None    # SharedContextStore 实例
    _context_manager: Any = None  # ContextManager 实例
    enable_context_management: bool = False
