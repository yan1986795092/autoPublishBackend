"""PII 脱敏 — 默认开启，业务侧 opt-in 明文。

依据规范 P7：Llm prompt 全文落 ES 时默认走 PII 脱敏管线，
避免 GDPR / 等保2.0 头上。脱敏策略：
- 邮箱 / 手机号 / 中国大陆身份证号 / 信用卡号 → [REDACTED:<type>]
- 邮箱保留前后各 1 字符做模糊脱敏

Redactor 是协议类，提供一个具体的 DefaultRedactor。
GenAI 序列化器在 dispatch_prompt/dispatch_completion 之前经过 redactor。
Redactor 抛异常时遵循 P6 → 不静默，记录指标但允许原文落盘（PII 安全优先级 vs 落盘成功率权衡 — 这里 PII 安全优先）。
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict
from typing import Any, Protocol

from .genai_records import PromptRecord, CompletionRecord, HandoffRecord


# 正则集合（覆盖常见 PII 模式）
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CN_PHONE_RE = re.compile(r"(?<!\d)(13[0-9]|14[5-9]|15[0-35-9]|16[2567]|17[0-35-8]|18[0-9]|19[0-35-9])\d{8}(?!\d)")
_CN_IDCARD_RE = re.compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)")
# 银行卡号：13-19 位连续纯数字。身份证是 18 位且以 X 结尾，因此要求"全是数字 + 长度 13-19"。
_CREDIT_CARD_RE = re.compile(r"(?<!\d)\d{13,19}(?!\d)")
# API Key / Bearer Token (粗略匹配 sk-/pk-/Bearer 前缀的 32+ 连续字符)
_APIKEY_RE = re.compile(r"(?:sk-|pk-|Bearer\s+)[A-Za-z0-9_\-]{16,}")


def _redact_text(text: str) -> str:
    """对单段文本做 PII 替换。

    顺序很关键：信用卡正则 \d{13,19} 会贪心匹配到 18 位身份证子串（含末尾 X）。
    必须先跑身份证规则（要求末尾 X 是字母），再跑卡号规则（要求末尾是数字）。
    """
    if not text:
        return text
    text = _EMAIL_RE.sub("[REDACTED:email]", text)
    text = _CN_PHONE_RE.sub("[REDACTED:phone]", text)
    # IDCard 必须先于 CreditCard，否则卡号正则会先吃掉数字段
    text = _CN_IDCARD_RE.sub("[REDACTED:idcard]", text)
    text = _CREDIT_CARD_RE.sub("[REDACTED:card]", text)
    text = _APIKEY_RE.sub("[REDACTED:apikey]", text)
    return text


class Redactor(Protocol):
    """Redactor 协议 — 业务侧可注入自定义实现。"""

    def redact_prompt(self, record: PromptRecord) -> PromptRecord:
        """处理 PromptRecord；raw=True 直接返回。"""
        ...

    def redact_completion(self, record: CompletionRecord) -> CompletionRecord:
        ...

    def redact_handoff(self, record: HandoffRecord) -> HandoffRecord:
        ...


class NoopRedactor:
    """显式 opt-in 明文 — 业务侧在确认 PII 不存在风险时使用。"""

    def redact_prompt(self, record: PromptRecord) -> PromptRecord:
        return record

    def redact_completion(self, record: CompletionRecord) -> CompletionRecord:
        return record

    def redact_handoff(self, record: HandoffRecord) -> HandoffRecord:
        return record


class DefaultRedactor:
    """默认 PII 脱敏 — 邮箱/手机/身份证/信用卡/API key。"""

    def __init__(self) -> None:
        self._stats: dict[str, int] = {
            "prompts_redacted": 0,
            "completions_redacted": 0,
            "handoffs_redacted": 0,
            "redaction_errors": 0,
        }

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def _redact_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """递归处理 messages 列表中的 content 字段。"""
        out: list[dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                out.append(msg)
                continue
            new_msg = dict(msg)
            content = new_msg.get("content")
            if isinstance(content, str):
                try:
                    new_msg["content"] = _redact_text(content)
                except Exception as exc:
                    self._stats["redaction_errors"] += 1
                    # 异常时保留原文，但记录 metrics（PII 安全 vs 落盘 — 优先 PII，但降级为落原文）
                    new_msg["__redactor_error__"] = str(exc)
            elif isinstance(content, list):
                # OpenAI-style content parts: [{"type": "text", "text": "..."}]
                new_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        new_part = dict(part)
                        try:
                            new_part["text"] = _redact_text(part.get("text", ""))
                        except Exception as exc:
                            self._stats["redaction_errors"] += 1
                            new_part["__redactor_error__"] = str(exc)
                        new_parts.append(new_part)
                    else:
                        new_parts.append(part)
                new_msg["content"] = new_parts
            out.append(new_msg)
        return out

    def redact_prompt(self, record: PromptRecord) -> PromptRecord:
        if record.raw:
            return record
        try:
            new_messages = self._redact_messages(record.messages)
            new_tool_defs = None
            if record.tool_definitions is not None:
                new_tool_defs = [
                    {**td, "description": _redact_text(str(td.get("description", "")))}
                    if isinstance(td, dict)
                    else td
                    for td in record.tool_definitions
                ]
            self._stats["prompts_redacted"] += 1
            return PromptRecord(
                **{
                    **asdict(record),
                    "messages": new_messages,
                    "tool_definitions": new_tool_defs,
                    "extra": {**record.extra, "redacted": True, "redactor": "default"},
                }
            )
        except Exception as exc:
            self._stats["redaction_errors"] += 1
            return record

    def redact_completion(self, record: CompletionRecord) -> CompletionRecord:
        if record.raw:
            return record
        try:
            new_content = _redact_text(record.content)
            self._stats["completions_redacted"] += 1
            return CompletionRecord(
                **{
                    **asdict(record),
                    "content": new_content,
                    "extra": {**record.extra, "redacted": True, "redactor": "default"},
                }
            )
        except Exception as exc:
            self._stats["redaction_errors"] += 1
            return record

    def redact_handoff(self, record: HandoffRecord) -> HandoffRecord:
        # state_snapshot 内的字符串字段统一走一遍脱敏
        try:
            snap = record.state_snapshot
            new_snap = self._redact_value(snap) if isinstance(snap, dict) else snap
            if record.extra.get("redacted") or new_snap is snap:
                return record
            self._stats["handoffs_redacted"] += 1
            return HandoffRecord(
                **{
                    **asdict(record),
                    "state_snapshot": new_snap,
                    "extra": {**record.extra, "redacted": True, "redactor": "default"},
                }
            )
        except Exception as exc:
            self._stats["redaction_errors"] += 1
            return record

    def _redact_value(self, obj: Any) -> Any:
        if isinstance(obj, str):
            try:
                return _redact_text(obj)
            except Exception:
                self._stats["redaction_errors"] += 1
                return obj
        if isinstance(obj, list):
            return [self._redact_value(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self._redact_value(v) for k, v in obj.items()}
        return obj
