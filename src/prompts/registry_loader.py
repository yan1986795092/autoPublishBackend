"""PromptRegistry YAML 序列化：将模板版本链导入/导出为 YAML 文件"""
from __future__ import annotations
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional

from .registry import PromptRegistry
from .registry_types import RegistryEntry


def load_from_file(registry: PromptRegistry, path: str) -> list[str]:
    """从 YAML 文件加载模板定义（含所有历史版本）。返回加载的模板名列表。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or not isinstance(data, dict):
        raise ValueError(f"Invalid prompt file format: {path}")

    name = data.get("name", path.stem)
    description = data.get("description", "")
    group = data.get("group", "general")
    role = data.get("role", "")

    loaded_names = []

    entry = RegistryEntry(
        name=name,
        description=description,
        group=group,
        role=role,
    )

    for vdata in data.get("versions", []):
        from .registry_types import RegistryTemplateVersion
        import hashlib

        template = vdata.get("template", "")
        variables = vdata.get("variables", [])
        import json
        var_str = json.dumps(sorted(json.dumps(v, sort_keys=True) for v in variables))
        content_hash = hashlib.sha256(
            (template.strip() + "|" + var_str).encode()
        ).hexdigest()[:12]

        try:
            date = datetime.fromisoformat(vdata["date"]) if isinstance(vdata.get("date"), str) else datetime.utcnow()
        except (ValueError, TypeError):
            date = datetime.utcnow()

        version = RegistryTemplateVersion(
            version=vdata.get("version", "v1.0"),
            template=template,
            hash=content_hash,
            date=date,
            status=vdata.get("status", "active"),
            changelog=vdata.get("changelog", ""),
            variables=variables,
            created_by=vdata.get("created_by", "system"),
            parent_hash=vdata.get("parent_hash"),
        )
        entry.versions.append(version)

    if entry.versions:
        registry._entries[name] = entry
        loaded_names.append(name)

    return loaded_names


def dump_to_file(registry: PromptRegistry, name: str, path: str) -> None:
    """将模板的完整版本链导出为 YAML 文件。"""
    entry = registry._entries.get(name)
    if not entry:
        raise KeyError(f"Template not found: {name}")

    data = {
        "name": entry.name,
        "description": entry.description,
        "group": entry.group,
        "role": entry.role,
        "versions": [],
    }

    for v in entry.versions:
        data["versions"].append({
            "version": v.version,
            "date": v.date.isoformat(),
            "status": v.status,
            "changelog": v.changelog,
            "template": v.template,
            "variables": v.variables,
            "created_by": v.created_by,
            "parent_hash": v.parent_hash,
        })

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
