"""Skill 解析、导入和存储 API"""
from __future__ import annotations
import json
import re
import logging
import uuid
import httpx
from typing import Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.database import get_db
from src.api.auth import get_current_user
from src.models import User, Skill, SkillCategory, SkillSource
from src.skills.builtins import BUILTIN_SKILLS, get_builtin_skill

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/skills", tags=["skills"])


# ===================== 请求/响应模型 =====================


class SkillCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    category: str = "writing"
    icon: str = "✍️"
    system_prompt: str = Field(..., min_length=1)
    user_prompt_template: str | None = None
    preset_plan: dict | None = None
    content_types: list[str] | None = None
    config_schema: dict | None = None
    tags: list[str] = []
    author: str | None = None
    version: str | None = None


class SkillUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    category: str | None = None
    icon: str | None = None
    system_prompt: str | None = None
    user_prompt_template: str | None = None
    preset_plan: dict | None = None
    content_types: list[str] | None = None
    config_schema: dict | None = None
    tags: list[str] | None = None
    is_enabled: bool | None = None


class SkillImportURLRequest(BaseModel):
    url: str = Field(..., description="SKILL.md 或 skill JSON 的原始 URL")
    name_override: str | None = None


class SkillImportFileRequest(BaseModel):
    file_content: str = Field(..., description="SKILL.md 或 skill JSON 的原始内容")
    file_name: str = Field(..., description="文件名，用于判断格式")


# ===================== Skill 解析（支持 SKILL.md 格式）=====================


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """解析 YAML frontmatter，返回 (metadata_dict, body_text)

    支持: key-value、内联数组 [a, b]、简单列表 - item、多行字符串 | >
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not match:
        return {}, content
    fm_text, body = match.groups()

    metadata: dict[str, Any] = {}
    lines = fm_text.split("\n")
    n = len(lines)
    i = 0

    while i < n:
        stripped = lines[i].strip()

        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # 列表项
        if stripped.startswith("- "):
            item = stripped[2:]
            # 追加到最后一个 list 值
            if metadata and isinstance(list(metadata.values())[-1], list):
                vals = list(metadata.values())
                for j in range(len(vals) - 1, -1, -1):
                    if isinstance(vals[j], list):
                        vals[j].append(item)
                        break
            i += 1
            continue

        # Key: value 解析
        colon_idx = stripped.index(":") if ":" in stripped else -1
        if colon_idx <= 0:
            i += 1
            continue

        key = stripped[:colon_idx].strip()
        raw_val = stripped[colon_idx + 1:].strip()

        if not raw_val or raw_val in ("null", "~"):
            metadata[key] = None
        elif raw_val.startswith('"') or raw_val.startswith("'"):
            metadata[key] = raw_val.strip('"\'')
        elif raw_val in ("true", "True"):
            metadata[key] = True
        elif raw_val in ("false", "False"):
            metadata[key] = False
        elif raw_val.startswith("[") and raw_val.endswith("]"):
            # 内联 JSON 数组
            try:
                metadata[key] = json.loads(raw_val)
            except Exception:
                metadata[key] = raw_val
        elif raw_val in ("|", ">", "|+", ">+", "|-", ">-"):
            # 多行字符串
            indent = len(lines[i]) - len(lines[i].lstrip())
            content_lines = []
            j = i + 1
            while j < n:
                next_line = lines[j]
                next_stripped = next_line.strip()
                if not next_stripped:
                    content_lines.append("")
                    j += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= indent:
                    break
                content_lines.append(next_stripped)
                j += 1
            metadata[key] = "\n".join(content_lines).strip()
            i = j - 1
        else:
            # 简单值（尝试数字）
            try:
                metadata[key] = int(raw_val)
            except ValueError:
                try:
                    metadata[key] = float(raw_val)
                except ValueError:
                    metadata[key] = raw_val

        i += 1

    return metadata, body.strip()


def _parse_skill_md(content: str, source_url: str | None = None) -> dict | None:
    """将 SKILL.md 内容解析为 Skill dict"""
    try:
        fm, body = _parse_frontmatter(content)

        # 提取 description 字段（YAML frontmatter 或文件顶部）
        description = fm.get("description", "")
        if not description:
            # 从文件顶部查找 description 行
            for line in body.split("\n")[:20]:
                line = line.strip()
                if line.startswith("description:") and len(line) > 15:
                    description = line.split("description:", 1)[1].strip().strip('"').strip("'")
                    break
                if line.startswith("## ") or line.startswith("# "):
                    break

        # 构建 skill 结构
        skill: dict[str, Any] = {
            "name": fm.get("name", "Unnamed Skill"),
            "description": description,
            "category": fm.get("category", "writing"),
            "icon": fm.get("icon", "✍️"),
            "system_prompt": body,  # 完整 markdown 内容作为 system prompt
            "user_prompt_template": fm.get("user_prompt_template"),
            "tags": fm.get("tags", []),
            "author": fm.get("author"),
            "version": fm.get("version"),
            "source": "imported",
        }

        # preset_plan 解析
        preset_str = fm.get("preset_plan", "")
        if isinstance(preset_str, str) and preset_str:
            try:
                skill["preset_plan"] = json.loads(preset_str)
            except Exception:
                parts = [p.strip() for p in preset_str.split(",") if p.strip()]
                if parts:
                    skill["preset_plan"] = {
                        "complexity": "medium",
                        "agent_sequence": parts,
                    }

        # content_types 解析
        ct = fm.get("content_types", [])
        if isinstance(ct, str):
            ct = [t.strip() for t in ct.split(",") if t.strip()]
        skill["content_types"] = ct or []

        return skill
    except Exception as e:
        logger.warning(f"解析 Skill markdown 失败: {e}")
        return None


def _parse_skill_json(content: str, source_url: str | None = None) -> dict | None:
    """解析 skill JSON 格式"""
    try:
        data = json.loads(content)

        # 支持 { "skill": {...} } 包装格式
        if "skill" in data and isinstance(data["skill"], dict):
            data = data["skill"]

        required = ["name", "system_prompt"]
        for field in required:
            if field not in data or not data[field]:
                return None

        return {
            "name": data.get("name"),
            "description": data.get("description", ""),
            "category": data.get("category", "writing"),
            "icon": data.get("icon", "✍️"),
            "system_prompt": data.get("system_prompt"),
            "user_prompt_template": data.get("user_prompt_template"),
            "preset_plan": data.get("preset_plan"),
            "content_types": data.get("content_types", []),
            "config_schema": data.get("config_schema"),
            "tags": data.get("tags", []),
            "author": data.get("author"),
            "version": data.get("version"),
            "source": "imported",
        }
    except Exception as e:
        logger.warning(f"解析 Skill JSON 失败: {e}")
        return None


def parse_skill(content: str, file_name: str | None = None, source_url: str | None = None) -> dict | None:
    """智能解析 skill 内容，自动识别格式"""
    # JSON 格式检测
    if file_name and file_name.endswith(".json"):
        return _parse_skill_json(content, source_url)

    content_stripped = content.strip()
    if content_stripped.startswith("{"):
        return _parse_skill_json(content_stripped, source_url)

    # YAML frontmatter 检测 -> SKILL.md 格式
    if "---" in content_stripped[:200]:
        return _parse_skill_md(content_stripped, source_url)

    return None


# ===================== 内置 Skill 列表 =====================


@router.get("/builtins")
async def list_builtins():
    """列出所有内置 Skill"""
    return {"success": True, "data": BUILTIN_SKILLS}


@router.get("/builtins/{skill_id}")
async def get_builtin(skill_id: str):
    """获取单个内置 Skill 详情"""
    skill = get_builtin_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="内置 Skill 不存在")
    return {"success": True, "data": skill}


# ===================== 用户 Skill CRUD =====================


@router.get("")
async def list_skills(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    category: str | None = None,
    source: str | None = None,
):
    """列出当前用户的所有 Skill（含元信息）"""
    query = db.query(Skill).filter(
        (Skill.user_id == current_user.id) | (Skill.source == SkillSource.BUILTIN)
    )
    if category:
        try:
            cat_enum = SkillCategory(category)
            query = query.filter(Skill.category == cat_enum)
        except Exception:
            pass
    if source:
        try:
            src_enum = SkillSource(source)
            query = query.filter(Skill.source == src_enum)
        except Exception:
            pass

    skills = query.order_by(Skill.use_count.desc(), Skill.created_at.desc()).all()
    return {
        "success": True,
        "data": [s.to_meta_dict() for s in skills],
    }


@router.get("/{skill_id}")
async def get_skill(
    skill_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取单个 Skill 详情"""
    # 先检查内置（无权限限制）
    builtin = get_builtin_skill(skill_id)
    if builtin:
        return {"success": True, "data": builtin}

    # 用户自定义/导入的 Skill 必须归属当前用户
    skill = db.query(Skill).filter(
        Skill.id == skill_id,
        Skill.user_id == current_user.id,
    ).first()
    if not skill:
        raise HTTPException(status_code=404, detail="Skill 不存在")
    return {"success": True, "data": skill.to_dict()}


@router.post("")
async def create_skill(
    request: SkillCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """创建新 Skill"""
    try:
        cat_enum = SkillCategory(request.category)
    except Exception:
        cat_enum = SkillCategory.CUSTOM

    skill = Skill(
        user_id=current_user.id,
        name=request.name,
        description=request.description,
        category=cat_enum,
        icon=request.icon,
        source=SkillSource.USER,
        system_prompt=request.system_prompt,
        user_prompt_template=request.user_prompt_template,
        preset_plan_json=json.dumps(request.preset_plan) if request.preset_plan else None,
        content_types_json=json.dumps(request.content_types) if request.content_types else None,
        config_schema_json=json.dumps(request.config_schema) if request.config_schema else None,
        tags=",".join(request.tags),
        author=request.author,
        version=request.version,
    )
    db.add(skill)
    db.commit()
    db.refresh(skill)
    return {"success": True, "data": skill.to_meta_dict()}


@router.put("/{skill_id}")
async def update_skill(
    skill_id: str,
    request: SkillUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """更新 Skill"""
    skill = db.query(Skill).filter(
        Skill.id == skill_id,
        Skill.user_id == current_user.id,
        Skill.source != SkillSource.BUILTIN,
    ).first()
    if not skill:
        raise HTTPException(status_code=404, detail="Skill 不存在或无权修改")

    if request.name is not None:
        skill.name = request.name
    if request.description is not None:
        skill.description = request.description
    if request.category is not None:
        try:
            skill.category = SkillCategory(request.category)
        except Exception:
            pass
    if request.icon is not None:
        skill.icon = request.icon
    if request.system_prompt is not None:
        skill.system_prompt = request.system_prompt
    if request.user_prompt_template is not None:
        skill.user_prompt_template = request.user_prompt_template
    if request.preset_plan is not None:
        skill.preset_plan_json = json.dumps(request.preset_plan) if request.preset_plan else None
    if request.content_types is not None:
        skill.content_types_json = json.dumps(request.content_types) if request.content_types else None
    if request.config_schema is not None:
        skill.config_schema_json = json.dumps(request.config_schema) if request.config_schema else None
    if request.tags is not None:
        skill.tags = ",".join(request.tags)
    if request.is_enabled is not None:
        skill.is_enabled = request.is_enabled

    db.commit()
    db.refresh(skill)
    return {"success": True, "data": skill.to_meta_dict()}


@router.delete("/{skill_id}")
async def delete_skill(
    skill_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """删除 Skill"""
    skill = db.query(Skill).filter(
        Skill.id == skill_id,
        Skill.user_id == current_user.id,
        Skill.source != SkillSource.BUILTIN,
    ).first()
    if not skill:
        raise HTTPException(status_code=404, detail="Skill 不存在或无权删除")

    db.delete(skill)
    db.commit()
    return {"success": True, "data": {"id": skill_id}}


# ===================== 导入（URL / 文件） =====================


@router.post("/import/url")
async def import_skill_from_url(
    request: SkillImportURLRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """从 URL 导入 Skill（支持 raw GitHub/Gist URL）"""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(request.url)
            response.raise_for_status()
            content = response.text
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"无法获取 URL 内容: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"请求失败: {str(e)}")

    parsed = parse_skill(content, source_url=request.url)
    if not parsed:
        raise HTTPException(
            status_code=400,
            detail="无法解析 Skill 内容。请确保是 SKILL.md 或 skill JSON 格式。"
        )

    if request.name_override:
        parsed["name"] = request.name_override

    skill = Skill(
        user_id=current_user.id,
        name=parsed["name"],
        description=parsed.get("description", ""),
        category=SkillCategory(parsed.get("category", "writing")),
        icon=parsed.get("icon", "✍️"),
        source=SkillSource.IMPORTED,
        system_prompt=parsed["system_prompt"],
        user_prompt_template=parsed.get("user_prompt_template"),
        preset_plan_json=json.dumps(parsed["preset_plan"]) if parsed.get("preset_plan") else None,
        content_types_json=json.dumps(parsed.get("content_types", [])) if parsed.get("content_types") else None,
        config_schema_json=json.dumps(parsed["config_schema"]) if parsed.get("config_schema") else None,
        tags=",".join(parsed.get("tags", [])),
        author=parsed.get("author") or _extract_author_from_url(request.url),
        version=parsed.get("version"),
    )
    db.add(skill)
    db.commit()
    db.refresh(skill)
    return {"success": True, "data": skill.to_meta_dict()}


@router.post("/import/file")
async def import_skill_from_file(
    request: SkillImportFileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """从文件内容导入 Skill"""
    parsed = parse_skill(request.file_content, file_name=request.file_name)
    if not parsed:
        raise HTTPException(
            status_code=400,
            detail="无法解析 Skill 内容。请确保是 SKILL.md 或 skill JSON 格式。"
        )

    skill = Skill(
        user_id=current_user.id,
        name=parsed["name"],
        description=parsed.get("description", ""),
        category=SkillCategory(parsed.get("category", "writing")),
        icon=parsed.get("icon", "✍️"),
        source=SkillSource.IMPORTED,
        system_prompt=parsed["system_prompt"],
        user_prompt_template=parsed.get("user_prompt_template"),
        preset_plan_json=json.dumps(parsed["preset_plan"]) if parsed.get("preset_plan") else None,
        content_types_json=json.dumps(parsed.get("content_types", [])) if parsed.get("content_types") else None,
        config_schema_json=json.dumps(parsed["config_schema"]) if parsed.get("config_schema") else None,
        tags=",".join(parsed.get("tags", [])),
        author=parsed.get("author"),
        version=parsed.get("version"),
    )
    db.add(skill)
    db.commit()
    db.refresh(skill)
    return {"success": True, "data": skill.to_meta_dict()}


@router.post("/import/preview")
async def preview_import(
    request: SkillImportFileRequest,
):
    """预览导入的 Skill 内容（不保存）"""
    parsed = parse_skill(request.file_content, file_name=request.file_name)
    if not parsed:
        raise HTTPException(
            status_code=400,
            detail="无法解析 Skill 内容"
        )

    # 截断 system_prompt 预览
    preview = {**parsed}
    if len(preview.get("system_prompt", "")) > 500:
        preview["system_prompt"] = preview["system_prompt"][:500] + "\n\n... (已截断)"

    return {"success": True, "data": preview}


# ===================== Skill 使用计数 =====================


@router.post("/{skill_id}/use")
async def record_skill_use(
    skill_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """记录 Skill 被使用（增加 use_count）"""
    # 内置 Skill 无归属，查询用户的自定义/导入 Skill
    skill = db.query(Skill).filter(
        Skill.id == skill_id,
        Skill.user_id == current_user.id,
    ).first()
    if not skill:
        # 内置 Skill 也计数（无 DB 记录则静默忽略计数）
        if get_builtin_skill(skill_id):
            return {"success": True}
        raise HTTPException(status_code=404, detail="Skill 不存在")
    skill.use_count = (skill.use_count or 0) + 1
    db.commit()
    return {"success": True}


# ===================== 工具函数 =====================


def _extract_author_from_url(url: str) -> str | None:
    """从 URL 中提取作者信息（如 GitHub 用户名）"""
    match = re.search(r"github\.com[/:]([^/]+)/", url)
    if match:
        return match.group(1)
    return None


def load_skill_full(skill_id: str, db: Session) -> dict | None:
    """加载 Skill 完整数据（内置 + 数据库）"""
    # 内置优先
    builtin = get_builtin_skill(skill_id)
    if builtin:
        return builtin

    skill = db.query(Skill).filter(
        Skill.id == skill_id,
        Skill.is_enabled == True,
    ).first()
    if skill:
        return skill.to_dict()

    return None
