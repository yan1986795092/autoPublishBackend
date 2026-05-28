"""
pytest 配置和 fixtures
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.main import app
from src.database import get_db
from src.models import Base, User
from src.core.security import create_access_token, hash_password


# 创建内存数据库用于测试
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 测试用户数据
TEST_USER = {
    "email": "test@example.com",
    "password": "TestPass123!",
    "nickname": "测试用户"
}


def override_get_db():
    """覆盖数据库依赖，使用测试数据库"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def client():
    """创建测试客户端"""
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def auth_client(client):
    """创建已认证的测试客户端（直接注入用户和 token，绕过 passlib/bcrypt 兼容性问题）"""
    db = TestingSessionLocal()
    try:
        user = User(
            email=TEST_USER["email"],
            password_hash=hash_password(TEST_USER["password"]),
            nickname=TEST_USER["nickname"],
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        token = create_access_token(data={"sub": user.id})
    finally:
        db.close()

    return client, token
