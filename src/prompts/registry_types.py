"""PromptRegistry 数据类型定义"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


VersionStatus = Literal["active", "deprecated", "archived"]


@dataclass
class RegistryTemplateVersion:
    """模板的一个版本"""
    version: str                          # "v2.3"
    template: str                         # 模板文本（含 {variable}）
    hash: str                             # SHA256(content + variables_def)[:12]
    date: datetime                        # 创建时间
    status: VersionStatus = "active"
    changelog: str = ""
    variables: list[dict] = field(default_factory=list)
    created_by: str = "system"            # "system" | "user" | "api"
    parent_hash: str | None = None

    @property
    def is_deprecated(self) -> bool:
        return self.status == "deprecated"

    @property
    def is_archived(self) -> bool:
        return self.status == "archived"


@dataclass
class RegistryEntry:
    """一个模板的完整注册信息"""
    name: str                        # "writing.task"
    description: str = ""
    group: str = "general"
    role: str = ""
    versions: list[RegistryTemplateVersion] = field(default_factory=list)

    @property
    def active_version(self) -> RegistryTemplateVersion | None:
        """返回 status 为 active 的版本，无 active 版本时返回 None。"""
        for v in self.versions:
            if v.status == "active":
                return v
        # 不静默降级到 versions[0]：无 active 意味着配置错误，返回 None 让调用方处理
        return None

    @property
    def latest_version(self) -> RegistryTemplateVersion | None:
        """返回 versions 列表的第一项（列表由调用方维护为最新优先的顺序）。"""
        return self.versions[0] if self.versions else None


@dataclass
class PromptDiff:
    """两个版本的结构化差异"""
    name: str
    version_a: str
    version_b: str
    lines_added: int
    lines_removed: int
    variables_changed: list[str]    # 变量名列表
    template_diff: str              # 行级 diff 文本
