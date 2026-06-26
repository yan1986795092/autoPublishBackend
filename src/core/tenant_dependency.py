"""租户作用域依赖

为 FastAPI 提供一个标准的依赖项，自动在请求结束时清理租户上下文。
防止请求之间的 ContextVar 污染（特别是同步依赖和 async 依赖混合时）。

用法：
    from src.core.tenant_dependency import tenant_scoped_user

    @router.get("/api/v1/contents")
    def list_contents(
        current_user: User = Depends(tenant_scoped_user),
        db: Session = Depends(get_db),
    ):
        # 这里 db.query(Content) 自动只返回 current_user 的数据
        ...
"""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from src.api.auth import get_current_user
from src.core.tenant import clear_tenant
from src.database import get_db
from src.models import User


def tenant_scoped_user(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """租户作用域依赖

    行为：
      1. 调用 get_current_user 验证 token 并设置 tenant（get_current_user 内部 set_tenant）
      2. 请求处理完成后清理 tenant ContextVar

    Yields:
        User: 当前已认证用户
    """
    try:
        yield current_user
    finally:
        clear_tenant()