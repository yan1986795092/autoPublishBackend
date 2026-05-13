from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
from typing import Optional

from src.database import get_db
from src.models import User
from src.schemas.auth import RegisterRequest, LoginRequest, RefreshTokenRequest
from src.core.security import hash_password, verify_password, create_access_token, create_refresh_token, decode_token
from src.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/auth", tags=["认证"])


def get_current_user(
    authorization: Optional[str] = Header(None, description="Bearer Token"),
    db: Session = Depends(get_db)
) -> User:
    """获取当前认证用户"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证信息",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization[7:]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 Token 类型",
        )
    user = db.query(User).filter(User.id == payload.get("sub")).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
        )
    return user


@router.post("/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """用户注册"""
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该邮箱已被注册"
        )

    user = User(
        email=request.email,
        password_hash=hash_password(request.password),
        nickname=request.nickname or request.email.split("@")[0],
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "success": True,
        "data": {
            "user_id": user.id,
            "email": user.email,
            "nickname": user.nickname,
        }
    }


@router.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """用户登录"""
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="邮箱或密码错误"
        )

    access_token = create_access_token(data={"sub": user.id})
    refresh_token = create_refresh_token(data={"sub": user.id})

    return {
        "success": True,
        "data": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.access_token_expire_minutes * 60,
            "user": {
                "id": user.id,
                "email": user.email,
                "nickname": user.nickname,
                "quota_daily": user.quota_daily,
                "quota_used_today": user.quota_used_today,
            }
        }
    }


@router.post("/refresh")
def refresh_token(request: RefreshTokenRequest, db: Session = Depends(get_db)):
    """刷新 Access Token"""
    payload = decode_token(request.refresh_token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh Token 无效或已过期"
        )
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 Token 类型"
        )

    user = db.query(User).filter(User.id == payload.get("sub")).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在"
        )

    new_access_token = create_access_token(data={"sub": user.id})
    return {
        "success": True,
        "data": {
            "access_token": new_access_token,
            "expires_in": settings.access_token_expire_minutes * 60,
        }
    }


@router.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    """获取当前用户信息"""
    return {
        "success": True,
        "data": {
            "id": current_user.id,
            "email": current_user.email,
            "nickname": current_user.nickname,
            "quota_daily": current_user.quota_daily,
            "quota_used_today": current_user.quota_used_today,
        }
    }
