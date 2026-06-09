from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional
from enum import Enum


class AgentType(str, Enum):
    MANAGER = "ManagerAgent"
    WRITING = "WritingAgent"
    SEO = "SEOAgent"
    DIAGRAM = "DiagramAgent"
    REVIEW = "ReviewAgent"
    PLATFORM = "PlatformAdapter"
    IMAGE = "ImageAgent"
    HOT_TOPIC = "HotTopicAgent"
    WEB_SEARCH = "WebSearch"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


@dataclass
class AgentTask:
    id: str
    agent_type: AgentType | str
    input: dict[str, Any]
    context: dict[str, Any] = field(default_factory=dict)
    parent_task_id: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 2
    can_parallel_with: list[str] = field(default_factory=list)
    retry_on_low_quality: bool = False
    priority: str = "medium"


@dataclass
class UsageInfo:
    tokens_used: int = 0
    estimated_cost: float = 0.0
    time_elapsed: float = 0.0
    model: str = ""


@dataclass
class AgentResult:
    task_id: str
    status: TaskStatus
    agent_type: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    usage: UsageInfo = field(default_factory=UsageInfo)
    agent_events: Optional[dict] = None
    # 上下文隔离：public 字段向用户展示和传递给下游 Agent，
    # internal 字段仅在编排层使用，不会泄漏到下游
    public_output: dict[str, Any] = field(default_factory=dict)
    internal: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, TaskStatus) else self.status
        # 移除 internal 字段，不序列化到外部
        if "internal" in d:
            del d["internal"]
        # 如果 public_output 有内容，用它作为主要 output
        if self.public_output and d.get("output") == {}:
            d["output"] = self.public_output
        return d

    def get_public_output(self) -> dict[str, Any]:
        """获取仅供下游 Agent 看到的输出内容（不含 internal 元数据）"""
        if self.public_output:
            return self.public_output
        # 从 output 中过滤掉内部字段
        internal_keys = {"task_id", "thinking_steps", "tokens_used", "_quality_score"}
        return {k: v for k, v in self.output.items() if k not in internal_keys}


@dataclass
class ExecutionPlan:
    """Manager Agent 制定的执行计划"""
    goal: str
    tasks: list[AgentTask]
    reasoning: str = ""
    estimated_steps: int = 0
    parallel_groups: list[list[str]] = field(default_factory=list)
    plan_method: str = "llm"


@dataclass
class ReActStep:
    """ReAct 循环中的一次 Think-Act-Observe 步骤"""
    step_index: int
    thought: str                        # LLM 思考：我当前要做什么
    action: str                        # 决定调用的工具或 peer_call
    action_input: dict                 # 工具参数
    observation: str                   # 观察结果
    raw_response: str = ""              # LLM 原始响应
    tool_name: Optional[str] = None     # 如果是工具调用，记录工具名
    peer_target: Optional[str] = None   # 如果是对等调用，记录目标 Agent


@dataclass
class PeerCallResult:
    """对等调用的结果"""
    target_agent: str
    call_id: str
    response: dict
    success: bool
    error: Optional[str] = None


@dataclass
class SubAgentConfig:
    """SubAgent 的配置参数"""
    max_react_steps: int = 8           # ReAct 最大迭代次数
    max_peer_calls: int = 5            # 单次任务中最多对等调用次数
    react_timeout: float = 60.0        # ReAct 循环超时（秒）
    enable_peer_collaboration: bool = True  # 是否启用对等协作
    peer_call_timeout: float = 30.0     # 单次 peer_call 超时
