import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("PYTEST_CURRENT_TEST", "1")

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.main import app
from src.models import Base, User
from src.core.security import create_access_token, hash_password
import src.database as database_module


SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

database_module.engine = engine
database_module.SessionLocal = TestingSessionLocal

TEST_USER = {
    "email": "test@example.com",
    "password": "TestPass123!",
    "nickname": "测试用户",
}


@pytest.fixture(scope="function")
def db_session():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    with TestClient(app) as test_client:
        yield test_client


def _create_user(db, email: str, password: str = TEST_USER["password"], nickname: str = TEST_USER["nickname"]) -> User:
    user = User(
        email=email,
        password_hash=hash_password(password),
        nickname=nickname,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture(scope="function")
def auth_client(client, db_session):
    user = _create_user(db_session, TEST_USER["email"])
    token = create_access_token(data={"sub": user.id})
    return client, token


@pytest.fixture(scope="function")
def two_users(db_session):
    user1 = _create_user(db_session, "user1@example.com", nickname="user1")
    user2 = _create_user(db_session, "user2@example.com", nickname="user2")
    return {
        "user1": user1,
        "user2": user2,
        "token1": create_access_token(data={"sub": user1.id}),
        "token2": create_access_token(data={"sub": user2.id}),
    }
