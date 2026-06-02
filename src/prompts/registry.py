"""PromptRegistry — 模板版本注册中心"""
from __future__ import annotations
import hashlib
import difflib
import json
import logging
from datetime import datetime, timezone
from typing import Literal

from .registry_types import (
    RegistryEntry,
    RegistryTemplateVersion,
    PromptDiff,
)
from .engine import RenderedPrompt

logger = logging.getLogger(__name__)


def _compute_hash(template: str, variables: list[dict]) -> str:
    """计算模板的内容 hash（SHA256 的前 12 位）。"""
    var_str = json.dumps(sorted(json.dumps(v, sort_keys=True) for v in variables))
    raw = template.strip() + "|" + var_str
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _next_version(current: str, major: bool = True) -> str:
    """自动递增版本号。"""
    prefix = ""
    if current.startswith("v"):
        prefix = "v"
        current = current[1:]
    try:
        major_num, minor_num = (current.split(".") + ["0"])[:2]
        if major:
            return f"{prefix}{int(major_num) + 1}.0"
        else:
            return f"{prefix}{major_num}.{int(minor_num) + 1}"
    except (ValueError, IndexError):
        return "v1.0"


class PromptRegistry:
    """模板版本注册中心。"""

    def __init__(self):
        self._entries: dict[str, RegistryEntry] = {}

    # ── 注册 / 新建版本 ──

    def register(self, name: str, template: str, *,
                 description: str = "",
                 variables: list[dict] | None = None,
                 changelog: str = "",
                 role: str = "",
                 group: str = "") -> RegistryEntry:
        """首次注册模板。如果已存在则转为 new_version。"""
        existing = self._entries.get(name)
        if existing:
            logger.info("Template %s already registered, creating new version", name)
            self.new_version(name, template, changelog=changelog, variables=variables)
            return existing

        content_hash = _compute_hash(template, variables or [])
        now = datetime.now(timezone.utc)
        version = RegistryTemplateVersion(
            version="v1.0",
            template=template.strip(),
            hash=content_hash,
            date=now,
            status="active",
            changelog=changelog or "首次注册",
            variables=variables or [],
            created_by="system",
        )
        entry = RegistryEntry(
            name=name,
            description=description,
            group=group,
            role=role,
            versions=[version],
        )
        self._entries[name] = entry
        logger.info("Template registered: %s (v1.0, hash=%s)", name, content_hash)
        return entry

    def new_version(self, name: str, template: str, *,
                    changelog: str = "",
                    variables: list[dict] | None = None
                    ) -> RegistryTemplateVersion | None:
        """为已有模板创建新版本。内容无变化时返回 None。"""
        entry = self._entries.get(name)
        if not entry:
            return None

        active_vars = entry.active_version.variables if entry.active_version else []
        content_hash = _compute_hash(template, variables or active_vars)
        latest = entry.versions[0]

        # 内容无变化则跳过
        if content_hash == latest.hash:
            return None

        now = datetime.now(timezone.utc)
        if latest.status == "active":
            latest.status = "deprecated"

        next_ver = _next_version(latest.version, major=True)
        new_ver = RegistryTemplateVersion(
            version=next_ver,
            template=template.strip(),
            hash=content_hash,
            date=now,
            status="active",
            changelog=changelog or f"更新至 {next_ver}",
            variables=variables or active_vars,
            created_by="system",
            parent_hash=latest.hash,
        )
        entry.versions.insert(0, new_ver)
        logger.info("Template %s: new version %s (hash=%s)", name, next_ver, content_hash)
        return new_ver

    # ── 查询 ──

    def get(self, name: str, *,
            pin: str | None = None,
            pin_hash: str | None = None) -> RegistryTemplateVersion | None:
        """获取指定版本的模板定义。不传 pin/pin_hash 返回 active 版本。"""
        entry = self._entries.get(name)
        if not entry:
            return None

        if pin_hash:
            for v in entry.versions:
                if v.hash == pin_hash:
                    return v
                if v.hash.startswith(pin_hash):
                    return v
            return None

        if pin:
            for v in entry.versions:
                if v.version == pin:
                    return v
            return None

        return entry.active_version

    def render(self, name: str, variables: dict | None = None, *,
               pin: str | None = None,
               pin_hash: str | None = None,
               strict: bool = False) -> RenderedPrompt:
        """渲染模板。"""
        version = self.get(name, pin=pin, pin_hash=pin_hash)
        if not version:
            if strict:
                raise KeyError(f"Template not found: {name}")
            return RenderedPrompt(
                text="", template_name=name,
                template_version="", variables_used={},
                errors=[f"Template not found: {name}"],
            )

        variables = variables or {}
        errors = []

        # 检查必填变量（variables 为 list[dict]）
        for var_def in version.variables:
            var_name = var_def["name"]
            required = var_def.get("required", False)
            val = variables.get(var_name)
            if required and (val is None or val == ""):
                msg = f"{name}: 缺少必填变量 '{var_name}'"
                errors.append(msg)
                if strict:
                    raise ValueError(msg)

        # 渲染
        try:
            text = version.template.format(**variables)
        except KeyError as e:
            msg = f"{name}: 模板变量 {e} 未提供"
            errors.append(msg)
            if strict:
                raise ValueError(msg)
            text = version.template
        except Exception as e:
            msg = f"{name}: 渲染异常: {e}"
            errors.append(msg)
            if strict:
                raise
            text = version.template

        return RenderedPrompt(
            text=text,
            template_name=name,
            template_version=version.version,
            variables_used=variables,
            errors=errors,
            template_hash=version.hash,
        )

    def list_versions(self, name: str) -> list[RegistryTemplateVersion]:
        """列出模板的所有版本，按时间降序。"""
        entry = self._entries.get(name)
        if not entry:
            return []
        return list(entry.versions)

    def list_templates(self, *, group: str = "", role: str = "",
                       status: str | None = None) -> list[RegistryEntry]:
        """列出注册的模板。"""
        results = list(self._entries.values())
        if group:
            results = [e for e in results if e.group == group]
        if role:
            results = [e for e in results if e.role == role]
        if status:
            results = [
                e for e in results
                if e.active_version and e.active_version.status == status
            ]
        return sorted(results, key=lambda e: e.name)

    def set_status(self, name: str, version_str: str,
                   status: Literal["active", "deprecated", "archived"]) -> None:
        """改变某个版本的状态。设为 active 时自动降级其他 active 版本。"""
        entry = self._entries.get(name)
        if not entry:
            raise KeyError(f"Template not found: {name}")

        target = None
        for v in entry.versions:
            if v.version == version_str:
                target = v
                break

        if not target:
            raise KeyError(f"Version {version_str} not found for {name}")

        if status == "active":
            for v in entry.versions:
                if v.status == "active" and v.version != version_str:
                    v.status = "deprecated"

        target.status = status

    def diff(self, name: str, version_a: str, version_b: str) -> PromptDiff:
        """比较两个版本的差异。"""
        va = self.get(name, pin=version_a)
        vb = self.get(name, pin=version_b)
        if not va or not vb:
            raise KeyError(f"Version not found: {name}@{version_a} or {version_b}")

        lines_a = va.template.splitlines()
        lines_b = vb.template.splitlines()

        diff_text = "\n".join(difflib.unified_diff(
            lines_a, lines_b,
            fromfile=f"{name}@{version_a}",
            tofile=f"{name}@{version_b}",
            lineterm="",
        ))

        added = sum(1 for l in diff_text.split("\n") if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_text.split("\n") if l.startswith("-") and not l.startswith("---"))

        vars_a = {v["name"]: v for v in va.variables} if va.variables else {}
        vars_b = {v["name"]: v for v in vb.variables} if vb.variables else {}
        changed_vars = list(set(list(vars_a.keys()) + list(vars_b.keys())))

        return PromptDiff(
            name=name,
            version_a=version_a,
            version_b=version_b,
            lines_added=added,
            lines_removed=removed,
            variables_changed=changed_vars,
            template_diff=diff_text,
        )
