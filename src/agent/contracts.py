"""Pydantic contracts for LLM-typed boundaries.

替换原 schemas.py 内 dataclass 风格的 WRITING_SCHEMA/REVIEW_SCHEMA/FIX_SCHEMA,
以及 _parse_plan_from_response 的 regex/json.loads 自由文本路径。
所有 LLM 输出的契约都从这里走, 这是 Layer 1(Schema 强约束)的基础。

为什么用 Pydantic 而不是继续走 dataclass:
- 自动生成 JSON Schema, 可直接喂给 LLM Gateway 的 strict mode
- 内置 ValidationError, 可注入 reask loop 作为错误反馈
- field_validator / model_validator 支持语义级校验(不只 schema 正确)

Reference:
- Anthropic tool_use: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- OpenAI strict mode: response_format={type:"json_schema", strict:true}
- Instructor 模式: max_retries + ValidationError 注入 messages
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Plan 层契约 ────────────────────────────────────────────────


class AgentTaskSpec(BaseModel):
    """LLM 输出的单个任务规约。

    必填字段校验: agent_type 必须在 registry 中存在; dependencies
    中的 id 必须能在同一份 plan 的其它 task 里找到, 或者为空(表示 root)。
    """

    id: str = Field(..., min_length=1, max_length=64)
    agent_type: str = Field(..., min_length=1)
    input: dict = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    can_parallel_with: list[str] = Field(default_factory=list)
    retry_on_low_quality: bool = False
    priority: Literal["low", "medium", "high"] = "medium"
    reasoning: str = ""

    @field_validator("agent_type")
    @classmethod
    def _agent_exists(cls, v: str) -> str:
        # 延迟导入避免循环
        from .registry import get
        if get(v) is None:
            available = []
            try:
                from .registry import all as _all_cards
                available = [c.name for c in _all_cards()]
            except Exception:
                pass
            raise ValueError(
                f"未知 agent_type '{v}'。可用 agent: {available}"
            )
        return v

    @field_validator("input")
    @classmethod
    def _input_must_be_dict(cls, v: dict) -> dict:
        if not isinstance(v, dict):
            raise ValueError("input 必须是 dict")
        return v


class PlannerOutput(BaseModel):
    """LLM Planner 必须严格按此 schema 输出。

    此 schema 同时可被 LLM Gateway 用作:
    - response_format (OpenAI strict mode)
    - tools[0].input_schema (Anthropic tool_use)

    注意: 实际 prompt 里仍要求 LLM 返回 JSON; 这里契约用来 parse + 校验 + reask,
    而非让 LLM 直接调用 tool。原因: Planner 产出是"计划", 不是"行动",
    没必要走 tool routing, 且 tool_use 兼容性问题较多。
    """

    goal: str = Field(..., min_length=1, max_length=500)
    tasks: list[AgentTaskSpec] = Field(..., min_length=1, max_length=20)
    reasoning: str = ""

    @model_validator(mode="after")
    def _check_dependency_refs(self) -> "PlannerOutput":
        """每个 dependencies 里的 id 必须能在本 plan 的其它 task 中找到;
        必须至少有一个 root task (无 dependency 的);
        不能存在环 (DFS 环检测)。

        FIX #3 顺序: 先环检测, 再 root 检测。因为 A→B→A 这种环里所有 task
        都有 dependency, root 检测会先报"无 root", 把环检测遮盖掉。
        """
        task_ids = {t.id for t in self.tasks}
        for t in self.tasks:
            for dep_id in t.dependencies:
                if dep_id not in task_ids:
                    raise ValueError(
                        f"task '{t.id}' 的 dependency '{dep_id}' "
                        f"不在本 plan 的 tasks 中"
                    )

        # FIX #3: DFS 环检测 (用颜色标记: 0=未访问, 1=访问中, 2=已完成)
        # 防止 LLM 输出 A→B→A 这种循环依赖导致 scheduler 死锁
        adj: dict[str, list[str]] = {t.id: list(t.dependencies) for t in self.tasks}
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {tid: WHITE for tid in task_ids}

        def _dfs(node: str, path: list[str]) -> None:
            if color[node] == GRAY:
                # 找到环
                cycle = " -> ".join(path[path.index(node):] + [node])
                raise ValueError(
                    f"plan 存在循环依赖: {cycle}"
                )
            if color[node] == BLACK:
                return
            color[node] = GRAY
            for nxt in adj.get(node, []):
                if nxt in color:
                    _dfs(nxt, path + [nxt])
            color[node] = BLACK

        for tid in task_ids:
            if color[tid] == WHITE:
                _dfs(tid, [tid])

        # 环检测通过后, 再确认有 root (无环+有 root 才能执行)
        roots = sum(1 for t in self.tasks if not t.dependencies)
        if roots == 0:
            # 无环但全有 dep: 这是不可能的状态(数学上), 视为数据错误
            raise ValueError(
                "plan 必须至少有一个 root task (无 dependency 的)"
            )

        return self


# ── Agent 输入输出契约 ────────────────────────────────────────


class WritingInput(BaseModel):
    topic: str = Field(..., min_length=1)
    content_type: str = "blog"
    tone: str | None = None
    audience: str | None = None
    platform: str | None = None
    style: str | None = None
    keywords: list[str] = Field(default_factory=list)


class WritingOutput(BaseModel):
    title: str
    body: str
    tags: list[str] = Field(default_factory=list)


class HotTopicInput(BaseModel):
    topic: str
    platform: str | None = None


class HotTopicOutput(BaseModel):
    content: str
    keywords: list[str] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)


class WebSearchInput(BaseModel):
    query: str
    count: int = 5


class WebSearchOutput(BaseModel):
    query: str
    results: list[dict] = Field(default_factory=list)
    summary: str = ""


class ReviewInput(BaseModel):
    content: str = Field(..., min_length=1)
    rules: list[str] = Field(default_factory=list)


class ReviewOutput(BaseModel):
    content: str
    passed: bool


class FixInput(BaseModel):
    content: str = Field(..., min_length=1)
    feedback: str | None = None


class FixOutput(BaseModel):
    content: str
    type: str = "blog"


# ── 默认可靠性配置 ────────────────────────────────────────────


DEFAULT_RELIABILITY_CONFIG = {
    "parse_max_attempts": 3,      # parse+reask 总尝试次数
    "max_retries": 3,             # FrozenPlan.RETRY 上限
    "max_patches": 2,             # FrozenPlan.PATCH 上限
    "max_replans": 1,             # FrozenPlan.REPLAN 上限
    "fallback_enabled": True,     # 失败时是否走 _build_fallback_plan
}