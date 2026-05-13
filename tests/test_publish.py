"""
发布接口测试
"""
import pytest


def get_auth_headers(client, email="pub@example.com", password="TestPass123!"):
    """辅助函数：获取认证头"""
    client.post("/api/v1/auth/register", json={
        "email": email, "password": password
    })
    resp = client.post("/api/v1/auth/login", json={
        "email": email, "password": password
    })
    token = resp.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestPublish:
    """发布模块测试"""

    def test_publish_without_auth(self, client):
        """测试未认证发布"""
        response = client.post("/api/v1/publish", json={
            "content_id": "some-id",
            "platform_code": "xiaohongshu"
        })
        assert response.status_code == 401

    def test_publish_nonexistent_content(self, client):
        """测试发布不存在的内容"""
        headers = get_auth_headers(client, "pubnonexist@example.com")
        response = client.post(
            "/api/v1/publish",
            headers=headers,
            json={
                "content_id": "nonexistent-id",
                "platform_code": "xiaohongshu",
                "publish_mode": "draft"
            }
        )
        assert response.status_code == 404

    def test_publish_draft_success(self, client):
        """测试成功保存草稿"""
        headers = get_auth_headers(client, "pubdraft@example.com")
        gen_resp = client.post("/api/v1/contents/generate", headers=headers, json={
            "topic": "测试主题"
        })
        content_id = gen_resp.json()["data"]["content_id"]

        response = client.post(
            "/api/v1/publish",
            headers=headers,
            json={
                "content_id": content_id,
                "platform_code": "xiaohongshu",
                "publish_mode": "draft"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["mode"] == "draft"
        assert "publish_record_id" in data["data"]

    def test_get_platforms(self, client):
        """测试获取平台列表"""
        response = client.get("/api/v1/platforms")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 3

        platform_codes = [p["code"] for p in data["data"]]
        assert "xiaohongshu" in platform_codes
        assert "wechat" in platform_codes
        assert "zhihu" in platform_codes


class TestTemplates:
    """模板模块测试"""

    def test_list_templates(self, client):
        """测试获取模板列表"""
        headers = get_auth_headers(client, "tmpl@example.com")
        response = client.get("/api/v1/templates", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
