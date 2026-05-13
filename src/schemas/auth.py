from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)
    nickname: Optional[str] = Field(None, max_length=50)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: str
    nickname: Optional[str] = None
    quota_daily: int
    quota_used_today: int

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class RegisterResponse(BaseModel):
    user_id: str
    email: str
    nickname: Optional[str] = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: Optional[str] = None


class APIResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[ErrorDetail] = None
