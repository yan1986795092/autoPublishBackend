from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import time
from typing import Any
import uuid


class EventType(str, Enum):
    AGENT_START = "agent_start"
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_FINISH = "agent_finish"
    ERROR = "error"


MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "glm-4": {"prompt": 0.001, "completion": 0.001},
    "glm-4-flash": {"prompt": 0.0001, "completion": 0.0001},
    "qwen-plus": {"prompt": 0.004, "completion": 0.012},
    "qwen-max": {"prompt": 0.02, "completion": 0.06},
}


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class AgentEvent:
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    run_id: str = ""
    step_index: int = 0
    parent_span_id: str = ""
    agent_name: str = ""
    event_type: str = ""
    level: str = "INFO"
    message: str = ""
    thought: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: str | None = None
    duration_ms: float | None = None
    token_usage: TokenUsage | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_stack: str | None = None
    llm_model: str | None = None
    llm_provider: str | None = None
    service: str = "autopublish-api"
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "@timestamp": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "step_index": self.step_index,
            "parent_span_id": self.parent_span_id,
            "agent_name": self.agent_name,
            "event_type": self.event_type,
            "level": self.level,
            "message": self.message,
            "service": self.service,
            "trace_id": self.trace_id,
        }
        optional_fields = {
            "thought": self.thought,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
            "duration_ms": self.duration_ms,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_stack": self.error_stack,
            "llm_model": self.llm_model,
            "llm_provider": self.llm_provider,
        }
        for key, value in optional_fields.items():
            if value is not None:
                result[key] = value
        if self.token_usage is not None:
            result["token_usage"] = {
                "prompt_tokens": self.token_usage.prompt_tokens,
                "completion_tokens": self.token_usage.completion_tokens,
                "total_tokens": self.token_usage.total_tokens,
                "cost_usd": self.token_usage.cost_usd,
            }
        return result


def calculate_cost_usd(token_usage: TokenUsage, model: str) -> float:
    pricing = MODEL_PRICING.get(model, {"prompt": 0.001, "completion": 0.003})
    prompt_cost = (token_usage.prompt_tokens / 1000) * pricing["prompt"]
    completion_cost = (token_usage.completion_tokens / 1000) * pricing["completion"]
    return round(prompt_cost + completion_cost, 6)
