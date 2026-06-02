"""PromptEngine — 模板注册、渲染、组合

核心能力：
  1. 注册模板（内建 + 用户自定义）
  2. 渲染模板（变量插值、校验、类型转换）
  3. 组合多段 prompt（system + instructions + context → messages）
  4. 版本追踪（记录使用了哪些模板和版本）
"""

from __future__ import annotations
import logging
import re
from typing import Optional
from dataclasses import dataclass, field

from .variables import VariableDef, VariableSet

logger = logging.getLogger(__name__)


# ── 模板数据类型 ──

@dataclass
class PromptTemplateDef:
    """一个注册的模板定义"""
    name: str                       # 唯一标识，如 "writing.task"
    template: str                   # 模板文本，使用 {variable} 占位
    description: str = ""
    version: str = "1.0"
    role: str = ""                  # 适用角色，如 "WritingAgent"
    group: str = "general"           # 分组，如 "writing" / "manager" / "review"
    variables: list[VariableDef] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ── 渲染结果 ──

@dataclass
class RenderedPrompt:
    """一次渲染的结果"""
    text: str
    template_name: str
    template_version: str
    variables_used: dict
    errors: list[str] = field(default_factory=list)
    template_hash: str | None = None   # 模板内容 hash（由 PromptRegistry.render 填充，engine.render 路径为空）

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


@dataclass
class ComposedMessages:
    """组合后的消息列表"""
    messages: list[dict] = field(default_factory=list)
    used_templates: list[str] = field(default_factory=list)
    total_chars: int = 0


# ── 模板引擎 ──

class PromptEngine:
    """模板引擎：注册 → 渲染 → 组合

    模板占位符使用 {variable_name} 语法。
    文本中的字面花括号需使用 {{ 和 }} 转义（Python str.format 约定）。
    """

    def __init__(self):
        self._templates: dict[str, PromptTemplateDef] = {}

    # ═══════════════════════════════════════════
    # 注册
    # ═══════════════════════════════════════════

    def register(self, template: PromptTemplateDef) -> None:
        """注册一个模板。同名模板会覆盖（允许自定义模板替换内建模板）。"""
        self._templates[template.name] = template
        logger.debug("Template registered: %s (v%s, role=%s)", template.name, template.version, template.role)

    def register_many(self, templates: list[PromptTemplateDef]) -> None:
        for t in templates:
            self.register(t)

    def get(self, name: str) -> Optional[PromptTemplateDef]:
        """按名称获取模板定义"""
        return self._templates.get(name)

    @property
    def count(self) -> int:
        return len(self._templates)

    def list(self, role: str = "", group: str = "") -> list[PromptTemplateDef]:
        """列出模板，可按角色/分组筛选"""
        results = list(self._templates.values())
        if role:
            results = [t for t in results if t.role == role]
        if group:
            results = [t for t in results if t.group == group]
        return results

    # ═══════════════════════════════════════════
    # 渲染
    # ═══════════════════════════════════════════

    def render(self, name: str, variables: dict | None = None,
               strict: bool = False) -> RenderedPrompt:
        """渲染一个命名的模板

        Args:
            name: 模板名称
            variables: 模板变量
            strict: True 时变量校验失败抛出异常

        Returns:
            渲染结果（包含文本、版本、错误信息）
        """
        template = self._templates.get(name)
        if not template:
            err = f"模板未注册: {name}"
            if strict:
                raise KeyError(err)
            logger.warning(err)
            return RenderedPrompt(
                text="", template_name=name,
                template_version="0.0", variables_used={},
                errors=[err],
            )

        variables = variables or {}

        # 变量校验和类型转换
        var_set = VariableSet(template.variables)
        resolved = var_set.resolve(variables)
        errors = []

        # 检查必填
        for var_def in template.variables:
            val = resolved.get(var_def.name)
            if var_def.required and (val is None or val == ""):
                msg = f"{template.name}: 缺少必填变量 '{var_def.name}'"
                errors.append(msg)
                if strict:
                    raise ValueError(msg)

        # 渲染
        try:
            text = template.template.format(**resolved)
        except KeyError as e:
            msg = f"{template.name}: 模板变量 {e} 未提供"
            errors.append(msg)
            if strict:
                raise ValueError(msg)
            text = template.template
        except Exception as e:
            msg = f"{template.name}: 渲染异常: {e}"
            errors.append(msg)
            if strict:
                raise
            text = template.template

        return RenderedPrompt(
            text=text,
            template_name=template.name,
            template_version=template.version,
            variables_used=resolved,
            errors=errors,
        )

    def render_text(self, template_text: str, variables: dict | None = None) -> str:
        """直接渲染一段模板文本（不经过注册表）

        这是轻量用法，适合临时构建 prompt 片段。
        """
        if not variables:
            return template_text
        try:
            return template_text.format(**(variables or {}))
        except KeyError:
            return template_text

    # ═══════════════════════════════════════════
    # 组合（多段 → messages）
    # ═══════════════════════════════════════════

    def compose(self, parts: list[tuple[str, str, dict | None]]) -> ComposedMessages:
        """组合多段 prompt 为 LLM messages

        Args:
            parts: [(role, template_name_or_text, variables_or_None), ...]
                role 取值 "system" / "user" / "assistant"
                template_name_or_text: 已注册的模板名 | 纯文本 | 空字符串

        Returns:
            组合后的消息列表 + 使用的模板清单
        """
        messages = []
        used_templates = []
        total_chars = 0

        for role, text_or_name, variables in parts:
            if not text_or_name:
                continue

            # 检查是否是已注册的模板名
            template = self._templates.get(text_or_name)
            if template:
                # 是已注册模板 → 渲染
                rendered = self.render(text_or_name, variables)
                content = rendered.text
                if rendered.template_name not in used_templates:
                    used_templates.append(rendered.template_name)
                if rendered.errors:
                    logger.warning("compose: %s", rendered.errors)
            else:
                # 是纯文本
                if variables:
                    content = text_or_name.format(**variables)
                else:
                    content = text_or_name

            if content:
                messages.append({"role": role, "content": content})
                total_chars += len(content)

        return ComposedMessages(
            messages=messages,
            used_templates=used_templates,
            total_chars=total_chars,
        )

    def compose_to_llm_messages(self, parts: list[tuple[str, str, dict | None]],
                                 langchain_format: bool = False) -> list:
        """组合并返回纯 messages 列表

        Args:
            langchain_format: True 时返回 LangChain Message 对象

        Returns:
            messages 列表（dict 或 LangChain Message）
        """
        composed = self.compose(parts)
        if not langchain_format:
            return composed.messages

        # 转为 LangChain 消息对象
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

    # ═══════════════════════════════════════════
    # DB 模板加载
    # ═══════════════════════════════════════════

    def load_user_templates(self, user_id: str = "") -> int:
        """从数据库加载用户自定义模板，覆盖内建模板

        Returns:
            加载的模板数
        """
        count = 0
        try:
            from src.database import SessionLocal
            from src.models import PromptTemplate

            with SessionLocal() as db:
                query = db.query(PromptTemplate).filter(PromptTemplate.is_public == True)
                if user_id:
                    query = query.filter(
                        (PromptTemplate.user_id == user_id) | (PromptTemplate.is_public == True)
                    )
                db_templates = query.all()
        except Exception as e:
            logger.warning("PromptEngine: 无法从 DB 加载模板: %s", e)
            return 0

        for t in db_templates:
            vars_list = []
            try:
                import json
                raw_vars = json.loads(t.variables_json) if t.variables_json else []
                for v in raw_vars:
                    vars_list.append(VariableDef(
                        name=v.get("name", "var"),
                        type=v.get("type", "string"),
                        required=v.get("required", False),
                        label=v.get("label", ""),
                        default=v.get("default"),
                        options=v.get("options"),
                    ))
            except Exception:
                pass

            # 用 DB 模板覆盖（DB 中 name 字段匹配引擎的模板名）
            if t.system_prompt:
                self.register(PromptTemplateDef(
                    name=f"db.{t.name}",
                    template=t.system_prompt,
                    description=t.description or t.name,
                    version="user",
                    variables=vars_list,
                    metadata={"db_id": t.id, "source": "user_template"},
                ))
                count += 1
            if t.user_template:
                self.register(PromptTemplateDef(
                    name=f"db.{t.name}.user",
                    template=t.user_template,
                    description=f"{t.name} (user)",
                    version="user",
                    variables=vars_list,
                    metadata={"db_id": t.id, "source": "user_template"},
                ))
                count += 1

        logger.info("PromptEngine: 从 DB 加载了 %d 个用户模板", count)
        return count


# ── 全局单例 ──
prompt_engine = PromptEngine()
