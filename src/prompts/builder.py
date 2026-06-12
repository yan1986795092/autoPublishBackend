"""PromptBuilder — 流式 Prompt 构建器

fluent API 风格，handler 中链式调用：
    prompt = PromptBuilder(engine)
        .system("writing.system")
        .instructions("writing.task", {"topic": ..., "content_type": ...})
        .context(search_result)
        .memory(memory_context)
        .build()
"""

from __future__ import annotations
from typing import Optional, Callable
import logging

from .engine import PromptEngine, prompt_engine as _default_engine, ComposedMessages

logger = logging.getLogger(__name__)

# 延迟导入钩子（由 BaseAgent 注入，避免循环依赖）
_on_build_complete: Optional[Callable] = None

def set_on_build_complete(callback: Optional[Callable]) -> None:
    global _on_build_complete
    _on_build_complete = callback


class PromptBuilder:
    """Prompt 构建器 — fluent API 构建 LLM messages

    典型的 handler 使用方式：
        result = (PromptBuilder()
            .system("writing.system", {"role": "写作专家"})
            .instructions("writing.task", {"topic": topic, "content_type": "blog"})
            .context(search_results)
            .memory(memory_context)
            .build())

        messages = result.messages           # [{role, content}, ...]
        templates = result.used_templates     # 使用了哪些模板
    """

    def __init__(self, engine: Optional[PromptEngine] = None):
        self._engine = engine or _default_engine
        self._parts: list[tuple[str, str, Optional[dict]]] = []
        self._raw_texts: list[tuple[str, str]] = []  # (role, text)

    # ── System ──

    def system(self, template_name: str, variables: Optional[dict] = None) -> "PromptBuilder":
        """添加 system prompt（使用已注册模板）"""
        self._parts.append(("system", template_name, variables))
        return self

    def system_text(self, text: str) -> "PromptBuilder":
        """直接添加 system prompt 文本"""
        if text:
            self._raw_texts.append(("system", text))
        return self

    # ── User ──

    def user(self, template_name: str, variables: Optional[dict] = None) -> "PromptBuilder":
        """添加 user prompt（使用已注册模板）"""
        self._parts.append(("user", template_name, variables))
        return self

    def user_text(self, text: str) -> "PromptBuilder":
        """直接添加 user prompt 文本"""
        if text:
            self._raw_texts.append(("user", text))
        return self

    # ── 语义快捷方式 ──

    def instructions(self, template_name: str,
                     variables: Optional[dict] = None) -> "PromptBuilder":
        """添加任务指令（作为 user message）"""
        self._parts.append(("user", template_name, variables))
        return self

    def context(self, text: str) -> "PromptBuilder":
        """添加上下文参考（作为 user message 尾部）"""
        if text:
            self._raw_texts.append(("user", text))
        return self

    def memory(self, text: str) -> "PromptBuilder":
        """添加记忆上下文（作为 user message 尾部）"""
        if text:
            self._raw_texts.append(("user", text))
        return self

    # ── Assistant ──

    def assistant(self, template_name: str,
                  variables: Optional[dict] = None) -> "PromptBuilder":
        """添加 assistant 示例（few-shot）"""
        self._parts.append(("assistant", template_name, variables))
        return self

    def assistant_text(self, text: str) -> "PromptBuilder":
        """直接添加 assistant 文本"""
        if text:
            self._raw_texts.append(("assistant", text))
        return self

    # ── 构建 ──

    def build(self) -> ComposedMessages:
        """构建最终的 messages 列表"""
        # 1. 渲染已注册模板
        composed = self._engine.compose(self._parts)

        # 2. 追加纯文本片段
        messages = list(composed.messages)
        for role, text in self._raw_texts:
            messages.append({"role": role, "content": text})

        # 3. 合并同 role 的连续消息
        merged = self._merge_consecutive(messages)
        result = ComposedMessages(
            messages=merged,
            used_templates=composed.used_templates,
            total_chars=sum(len(m["content"]) for m in merged),
        )

        # ── 注册到 SnapshotCollector（如当前 context 有 collector）──
        if _on_build_complete is not None:
            try:
                _on_build_complete(composed=result, messages=merged)
            except Exception as e:
                logger.warning("PromptBuilder 构建钩子执行失败: %s", e)

        return result

    def build_langchain(self):
        """构建为 LangChain Message 列表"""
        composed = self.build()
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        ROLE_MAP = {
            "system": SystemMessage,
            "user": HumanMessage,
            "assistant": AIMessage,
        }
        return [
            ROLE_MAP.get(m["role"], HumanMessage)(content=m["content"])
            for m in composed.messages
        ]

    @staticmethod
    def _merge_consecutive(messages: list[dict]) -> list[dict]:
        """合并连续相同 role 的 message"""
        if not messages:
            return []

        merged = [dict(messages[0])]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(dict(msg))
        return merged

    # ── 重置 ──

    def reset(self) -> "PromptBuilder":
        self._parts = []
        self._raw_texts = []
        return self
