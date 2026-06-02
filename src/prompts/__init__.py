"""Prompt 模板模块 — 模板引擎 + 内建模板 + PromptBuilder

提供从注册、渲染到组合的完整流程，所有 handler 通过 PromptEngine 构建 prompt。

使用方式：
    from src.prompts import prompt_engine, PromptBuilder

    # 方式一：Builder 流式构建
    result = (PromptBuilder()
        .system("system.base")
        .instructions("writing.task", {"topic": "AI编程", "content_type": "blog"})
        .context(search_text)
        .memory(memory_text)
        .build())
    messages = result.messages

    # 方式二：直接渲染
    rendered = prompt_engine.render("writing.task", {"topic": "..."})
    text = rendered.text
"""

from .engine import PromptEngine, prompt_engine, PromptTemplateDef, RenderedPrompt, ComposedMessages
from .builder import PromptBuilder
from .variables import VariableDef, VariableSet
from .builtins import register_all as _register_all
from .registry import PromptRegistry
from .registry_types import RegistryEntry, RegistryTemplateVersion, PromptDiff

# ── 启动时自动注册内建模板 ──
_registered_count = _register_all()

# ── 向后兼容：保留原有的常量导出 ──
from .builtins.writing import CONTENT_TYPE_PROMPTS, STYLE_GUIDES
from .builtins.system import templates as _sys_templates
from .builtins.manager import templates as _mgr_templates

__all__ = [
    "prompt_engine",
    "PromptEngine",
    "PromptBuilder",
    "PromptTemplateDef",
    "RenderedPrompt",
    "ComposedMessages",
    "VariableDef",
    "VariableSet",
    "CONTENT_TYPE_PROMPTS",
    "STYLE_GUIDES",
    "PromptRegistry",
    "RegistryEntry",
    "RegistryTemplateVersion",
    "PromptDiff",
]
