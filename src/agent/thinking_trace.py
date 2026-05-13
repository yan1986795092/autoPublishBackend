from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Literal, Optional
import time
import uuid


@dataclass
class ThinkingStep:
    step_number: int
    agent: str
    action: str
    reasoning: str = ""
    input_summary: str = ""
    output_summary: str = ""
    tokens_used: int = 0
    time_elapsed: float = 0.0
    status: Literal["running", "done", "failed"] = "running"
    prompt_used: str = ""       # 实际使用的完整 Prompt
    response_used: str = ""     # 实际响应内容摘要

    def complete(self, output_summary: str = "", tokens_used: int = 0, elapsed: float = 0,
                 prompt_used: str = "", response_used: str = "") -> None:
        self.status = "done"
        self.output_summary = output_summary or self.output_summary
        self.tokens_used = tokens_used or self.tokens_used
        self.time_elapsed = elapsed or self.time_elapsed
        self.prompt_used = prompt_used or self.prompt_used
        self.response_used = response_used or self.response_used

    def fail(self, error: str = "") -> None:
        self.status = "failed"
        self.output_summary = error or self.output_summary

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class ThinkingTrace:
    """记录 Agent 的完整思考链路，用于可观测性展示"""

    def __init__(self, task_id: str = ""):
        self.task_id = task_id or str(uuid.uuid4())
        self.steps: list[ThinkingStep] = []
        self._start_time: float = 0.0
        self._step_start_times: dict[int, float] = {}
        self._current_step: int = 0

    def begin(self) -> None:
        self._start_time = time.time()

    @property
    def total_elapsed(self) -> float:
        if self._start_time == 0:
            return 0.0
        return round(time.time() - self._start_time, 2)

    def add_step(self, agent: str, action: str, reasoning: str = "", input_summary: str = "") -> int:
        self._current_step += 1
        step = ThinkingStep(
            step_number=self._current_step,
            agent=agent,
            action=action,
            reasoning=reasoning,
            input_summary=input_summary,
        )
        self.steps.append(step)
        self._step_start_times[self._current_step - 1] = time.time()
        return self._current_step - 1  # index

    def complete_step(self, step_index: int, output_summary: str = "", tokens_used: int = 0) -> None:
        if 0 <= step_index < len(self.steps):
            step = self.steps[step_index]
            now = time.time()
            step_start = self._step_start_times.get(step_index, self._start_time)
            elapsed = round(now - step_start, 2) if step_start else 0
            step.complete(output_summary=output_summary, tokens_used=tokens_used, elapsed=elapsed)

    def fail_step(self, step_index: int, error: str = "") -> None:
        if 0 <= step_index < len(self.steps):
            self.steps[step_index].fail(error=error)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "total_elapsed": self.total_elapsed,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ThinkingTrace:
        trace = cls(task_id=data.get("task_id", ""))
        trace.steps = [ThinkingStep(**s) for s in data.get("steps", [])]
        return trace
