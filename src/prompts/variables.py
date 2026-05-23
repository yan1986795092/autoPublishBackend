"""变量系统 — 模板变量的定义、校验和解析"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VariableDef:
    """一个模板变量的定义"""
    name: str
    type: str = "string"           # string | int | list | dict | bool
    required: bool = True
    label: str = ""
    default: Any = None
    description: str = ""
    options: Optional[list[str]] = None   # 枚举值限制

    def validate(self, value: Any) -> str | None:
        """校验变量值，返回 None 表示通过，返回错误消息表示不通过"""
        if value is None or value == "":
            if self.required:
                return f"缺少必填变量: {self.name}"
            return None

        if self.type == "int":
            try:
                int(value)
            except (ValueError, TypeError):
                return f"{self.name} 应为整数类型"
        elif self.type == "list":
            if not isinstance(value, (list, tuple)):
                return f"{self.name} 应为列表类型"

        if self.options and str(value) not in self.options:
            return f"{self.name} 取值需在 [{', '.join(self.options)}] 中"

        return None

    def coerce(self, value: Any) -> Any:
        """类型转换"""
        if value is None and not self.required:
            return self.default
        if self.type == "int":
            return int(value) if value is not None else self.default
        if self.type == "bool":
            if isinstance(value, bool):
                return value
            return str(value).lower() in ("true", "1", "yes") if value is not None else self.default
        if self.type == "list":
            return list(value) if value is not None else (self.default or [])
        return value if value is not None else self.default


class VariableSet:
    """一组已解析的变量，可直接用于模板渲染"""

    def __init__(self, definitions: list[VariableDef]):
        self._defs = {d.name: d for d in definitions}

    def resolve(self, provided: dict) -> dict:
        """将外部传入的变量与定义合并，填充默认值，校验类型"""
        resolved = {}
        errors = []

        for name, var_def in self._defs.items():
            raw = provided.get(name)
            error = var_def.validate(raw)
            if error:
                errors.append(error)
                resolved[name] = var_def.coerce(raw)
            else:
                resolved[name] = var_def.coerce(raw)

        # 额外传入但未定义的变量也保留（宽松模式）
        for k, v in provided.items():
            if k not in resolved:
                resolved[k] = v

        return resolved

    @property
    def required_names(self) -> list[str]:
        return [d.name for d in self._defs if d.required]

    @property
    def all_names(self) -> list[str]:
        return [d.name for d in self._defs]

    def to_schema(self) -> list[dict]:
        return [
            {
                "name": d.name,
                "type": d.type,
                "required": d.required,
                "label": d.label,
                "default": d.default,
                "description": d.description,
                "options": d.options,
            }
            for d in self._defs
        ]
