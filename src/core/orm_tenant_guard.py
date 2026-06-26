"""ORM 租户守卫 — 系统级强制 user_id 隔离

通过 SQLAlchemy 2.0 的 with_loader_criteria + do_orm_execute 钩子，
在 ORM 查询层自动注入 user_id 过滤，让"忘记加 user_id 过滤"从 bug 变成不可能。

SQLAlchemy 2.0 推荐用 with_loader_criteria 而不是直接改 statement.where()，
因为：
  - 自动传播到 relationship loader（lazy load / selectinload 等）
  - 自动应用到子查询
  - 自动处理 JOIN 时的别名

两层防护：
  1. do_orm_execute 钩子：所有 ORM SELECT 自动追加 WHERE user_id = :tenant_id
  2. before_flush 钩子：所有 INSERT/UPDATE 自动校验/填充 user_id

系统级任务（迁移脚本、后台任务）应使用 run_as_system() 临时禁用。
"""

from __future__ import annotations

import os

from sqlalchemy import event
from sqlalchemy.orm import Session, with_loader_criteria

from src.core.tenant import get_tenant
from src.models import (
    AgentTask,
    ChatConversation,
    Content,
    Persona,
    PlatformAccount,
    Preset,
    PromptTemplate,
    Skill,
)


# 环境变量开关（紧急回滚用：ENABLE_TENANT_GUARD=false）
def _is_enabled() -> bool:
    return os.getenv("ENABLE_TENANT_GUARD", "true").lower() in ("true", "1", "yes")


# 需要按租户隔离的模型白名单
TENANT_SCOPED_MODELS = frozenset({
    Content,             # 内容
    Preset,              # 用户预设
    Persona,             # 人设
    PlatformAccount,     # 平台账号
    PromptTemplate,      # 提示词模板
    Skill,               # 技能
    ChatConversation,    # 对话
    AgentTask,           # Agent 任务
})


def _make_criteria_for_model(model, tenant_id: str):
    """为单个模型生成 with_loader_criteria 过滤条件"""
    return with_loader_criteria(
        model,
        lambda cls: cls.user_id == tenant_id,
        include_aliases=True,
    )


@event.listens_for(Session, "do_orm_execute")
def inject_tenant_filter(state):
    """ORM SELECT 自动注入 WHERE user_id = :tenant_id

    SQLAlchemy 2.0 标准做法：用 with_loader_criteria 替代直接修改 where。
    优势：自动传播到 relationship loaders / subqueries。
    """
    if not _is_enabled():
        return

    if not state.is_select:
        return

    # column load 和 relationship load 已被顶层 query 自动处理，跳过
    if state.is_column_load or state.is_relationship_load:
        return

    tenant_id = get_tenant()
    if tenant_id is None:
        # 系统级查询允许通过
        return

    # 通过 statement 的 column_descriptions 收集模型
    seen_models = set()
    if hasattr(state.statement, "column_descriptions"):
        for desc in state.statement.column_descriptions:
            entity = desc.get("entity")
            if entity is None or entity in seen_models:
                continue
            seen_models.add(entity)

            if entity in TENANT_SCOPED_MODELS:
                state.statement = state.statement.options(
                    _make_criteria_for_model(entity, tenant_id)
                )


@event.listens_for(Session, "before_flush")
def validate_tenant_on_write(session, flush_context, instances):
    """写入时校验 user_id 与 tenant 一致

    行为：
      - tenant 已设置 + 新对象 user_id 为空：自动填充
      - tenant 已设置 + 新对象 user_id 不匹配：拒绝
      - tenant 未设置（系统级）：放行
    """
    if not _is_enabled():
        return

    tenant_id = get_tenant()
    if tenant_id is None:
        return

    for obj in session.new | session.dirty:
        model = type(obj)
        if model not in TENANT_SCOPED_MODELS:
            continue

        obj_user_id = getattr(obj, "user_id", None)

        if obj in session.new:
            if obj_user_id is None:
                obj.user_id = tenant_id
            elif obj_user_id != tenant_id:
                raise PermissionError(
                    f"[TENANT_GUARD] {model.__name__} 写入时 user_id "
                    f"({obj_user_id}) 与当前 tenant ({tenant_id}) 不一致，"
                    f"禁止跨用户写入"
                )
        else:
            if obj_user_id is not None and obj_user_id != tenant_id:
                raise PermissionError(
                    f"[TENANT_GUARD] {model.__name__} 更新时 user_id "
                    f"({obj_user_id}) 与当前 tenant ({tenant_id}) 不一致，"
                    f"禁止跨用户修改"
                )