"""
认证接口测试
"""
import pytest


class TestAuth:
    """认证模块测试"""

    def test_register_success(self, client):
        """测试成功注册"""
        response = client.post("/api/v1/auth/register", json={
            "email": "test2@example.com",
            "password": "TestPass123!",
            "nickname": "测试用户"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["user"]["email"] == "test2@example.com"

    def test_register_duplicate_email(self, client):
        """测试重复邮箱注册"""
        response = client.post("/api/v1/auth/register", json={
            "email": "dup@example.com",
            "password": "TestPass123!",
        })
        assert response.status_code == 200
        response = client.post("/api/v1/auth/register", json={
            "email": "dup@example.com",
            "password": "TestPass123!",
        })
        assert response.status_code == 400

    def test_register_invalid_email(self, client):
        """测试无效邮箱"""
        response = client.post("/api/v1/auth/register", json={
            "email": "invalid-email",
            "password": "TestPass123!",
        })
        assert response.status_code == 422

    def test_register_short_password(self, client):
        """测试密码过短"""
        response = client.post("/api/v1/auth/register", json={
            "email": "test@example.com",
            "password": "123",
        })
        assert response.status_code == 422

    def test_login_success(self, client):
        """测试成功登录"""
        client.post("/api/v1/auth/register", json={
            "email": "login@example.com",
            "password": "TestPass123!",
        })
        response = client.post("/api/v1/auth/login", json={
            "email": "login@example.com",
            "password": "TestPass123!"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "access_token" in data["data"]

    def test_login_wrong_password(self, client):
        """测试错误密码"""
        client.post("/api/v1/auth/register", json={
            "email": "wrong@example.com",
            "password": "TestPass123!",
        })
        response = client.post("/api/v1/auth/login", json={
            "email": "wrong@example.com",
            "password": "WrongPassword123!"
        })
        assert response.status_code == 401

    def test_get_me(self, client):
        """测试获取当前用户信息"""
        client.post("/api/v1/auth/register", json={
            "email": "me@example.com",
            "password": "TestPass123!",
        })
        login_resp = client.post("/api/v1/auth/login", json={
            "email": "me@example.com",
            "password": "TestPass123!"
        })
        token = login_resp.json()["data"]["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["email"] == "me@example.com"

    def test_get_me_no_token(self, client):
        """测试未提供 Token"""
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401

    def test_get_me_invalid_token(self, client):
        """测试无效 Token"""
        response = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer invalid_token"})
        assert response.status_code == 401
