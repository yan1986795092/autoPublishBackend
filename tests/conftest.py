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
from src.models import Base


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
    """创建已认证的测试客户端"""
    # 注册用户
    reg_response = client.post("/api/v1/auth/register", json=TEST_USER)
    assert reg_response.status_code == 200, f"Register failed: {reg_response.json()}"

    # 登录获取 token
    login_response = client.post("/api/v1/auth/login", json={
        "email": TEST_USER["email"],
        "password": TEST_USER["password"]
    })
    assert login_response.status_code == 200, f"Login failed: {login_response.json()}"

    token = login_response.json()["data"]["access_token"]
    return client, token
