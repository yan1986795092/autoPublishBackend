"""执行快照数据模型"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PromptMessageSnapshot:
    """单条 prompt message 的快照"""
    role: str                             # "system" | "user" | "assistant"
    template_name: str | None = None      # "system.base"
    template_version: str | None = None   # "v2.0"
    template_hash: str | None = None      # "abc123"
    source: str = "raw_text"              # "template" | "raw_text" | "memory" | "tool_result"
    resolved_content: str = ""
    variables: dict | None = None
    truncated: bool = False


@dataclass
class ContextSnapshot:
    """执行上下文快照"""
    profile_snapshot: dict | None = None           # AgentProfile.to_dict()
    memory_ids: list[str] = field(default_factory=list)
    memory_text_preview: str = ""
    search_results_hash: str | None = None
    search_query: str | None = None
    tool_calls: list[dict] = field(default_factory=list)  # [{tool, input_summary, output_summary}]
    extra_context_keys: list[str] = field(default_factory=list)


@dataclass
class LlmRequestSnapshot:
    """LLM 请求参数快照"""
    model: str
    params: dict | None = None
    messages_count: int = 0
    total_prompt_chars: int = 0


@dataclass
class LlmResponseSnapshot:
    """LLM 响应快照"""
    content: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0


@dataclass
class ExecutionSnapshot:
    """一次 Agent 执行的完整快照"""
    snapshot_id: str                              # "snap_<ulid>"
    agent_name: str                               # "WritingAgent"
    timestamp: datetime
    duration_ms: int = 0

    task_id: str | None = None
    parent_snapshot_id: str | None = None         # rerun 追踪

    prompt_messages: list[PromptMessageSnapshot] = field(default_factory=list)
    context: ContextSnapshot | None = None
    llm_request: LlmRequestSnapshot | None = None
    llm_response: LlmResponseSnapshot | None = None

    result_summary: str = ""
    result: dict | None = None

    tags: list[str] = field(default_factory=list)
    captured_hash: str = ""                       # 整个 snapshot 的 hash（防篡改）

    def to_dict(self) -> dict:
        """序列化为 JSON-compatible dict。"""
        import hashlib, json
        d = {
            "snapshot_id": self.snapshot_id,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "task_id": self.task_id,
            "parent_snapshot_id": self.parent_snapshot_id,
            "prompt_messages": [
                {k: v for k, v in m.__dict__.items() if v is not None or k in ("role", "source")}
                for m in self.prompt_messages
            ],
            "context": self.context.__dict__ if self.context else None,
            "llm_request": self.llm_request.__dict__ if self.llm_request else None,
            "llm_response": self.llm_response.__dict__ if self.llm_response else None,
            "result_summary": self.result_summary,
            "result": self.result,
            "tags": self.tags,
        }
        d["captured_hash"] = hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionSnapshot:
        """从 dict 反序列化。"""
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            from datetime import datetime
            timestamp = datetime.fromisoformat(timestamp)

        prompt_msgs = []
        for md in data.get("prompt_messages", []):
            prompt_msgs.append(PromptMessageSnapshot(**md))

        ctx = None
        if data.get("context"):
            ctx = ContextSnapshot(**data["context"])

        llm_req = None
        if data.get("llm_request"):
            llm_req = LlmRequestSnapshot(**data["llm_request"])

        llm_resp = None
        if data.get("llm_response"):
            llm_resp = LlmResponseSnapshot(**data["llm_response"])

        return cls(
            snapshot_id=data["snapshot_id"],
            agent_name=data["agent_name"],
            timestamp=timestamp,
            duration_ms=data.get("duration_ms", 0),
            task_id=data.get("task_id"),
            parent_snapshot_id=data.get("parent_snapshot_id"),
            prompt_messages=prompt_msgs,
            context=ctx,
            llm_request=llm_req,
            llm_response=llm_resp,
            result_summary=data.get("result_summary", ""),
            result=data.get("result"),
            tags=data.get("tags", []),
            captured_hash=data.get("captured_hash", ""),
        )
