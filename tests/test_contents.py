"""
内容接口测试
"""
import pytest


def get_auth_headers(client, email="gen@example.com", password="TestPass123!"):
    """辅助函数：获取认证头"""
    client.post("/api/v1/auth/register", json={
        "email": email, "password": password
    })
    resp = client.post("/api/v1/auth/login", json={
        "email": email, "password": password
    })
    token = resp.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestContents:
    """内容管理模块测试"""

    def test_generate_content_without_auth(self, client):
        """测试未认证生成内容"""
        response = client.post("/api/v1/contents/generate", json={
            "topic": "测试主题"
        })
        assert response.status_code == 401

    def test_generate_content_success(self, client):
        """测试成功生成内容"""
        headers = get_auth_headers(client, "gen1@example.com")
        response = client.post(
            "/api/v1/contents/generate",
            headers=headers,
            json={
                "topic": "职场沟通技巧",
                "platform_code": "xiaohongshu",
                "parameters": {
                    "length": "medium",
                    "creativity": 0.7,
                    "tone": "friendly"
                }
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "content_id" in data["data"]
        assert "title" in data["data"]

    def test_generate_content_empty_topic(self, client):
        """测试空主题"""
        headers = get_auth_headers(client, "gen2@example.com")
        response = client.post(
            "/api/v1/contents/generate",
            headers=headers,
            json={"topic": ""}
        )
        assert response.status_code == 422

    def test_list_contents(self, client):
        """测试获取内容列表"""
        headers = get_auth_headers(client, "list@example.com")
        client.post("/api/v1/contents/generate", headers=headers, json={
            "topic": "测试主题"
        })

        response = client.get("/api/v1/contents", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["items"]) >= 1

    def test_list_contents_pagination(self, client):
        """测试内容列表分页"""
        headers = get_auth_headers(client, "page@example.com")
        response = client.get(
            "/api/v1/contents",
            headers=headers,
            params={"page": 1, "page_size": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["pagination"]["page"] == 1

    def test_get_content_detail(self, client):
        """测试获取内容详情"""
        headers = get_auth_headers(client, "detail@example.com")
        gen_resp = client.post("/api/v1/contents/generate", headers=headers, json={
            "topic": "测试主题"
        })
        content_id = gen_resp.json()["data"]["content_id"]

        response = client.get(f"/api/v1/contents/{content_id}", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["id"] == content_id
        assert "versions" in data["data"]

    def test_get_nonexistent_content(self, client):
        """测试获取不存在的内容"""
        headers = get_auth_headers(client, "nonexist@example.com")
        response = client.get("/api/v1/contents/nonexistent-id", headers=headers)
        assert response.status_code == 404

    def test_update_content(self, client):
        """测试更新内容"""
        headers = get_auth_headers(client, "update@example.com")
        gen_resp = client.post("/api/v1/contents/generate", headers=headers, json={
            "topic": "测试主题"
        })
        content_id = gen_resp.json()["data"]["content_id"]

        response = client.put(
            f"/api/v1/contents/{content_id}",
            headers=headers,
            json={"title": "更新后的标题", "change_summary": "测试更新"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["version_number"] == 2

    def test_delete_content(self, client):
        """测试删除内容（软删除）"""
        headers = get_auth_headers(client, "delete@example.com")
        gen_resp = client.post("/api/v1/contents/generate", headers=headers, json={
            "topic": "测试主题"
        })
        content_id = gen_resp.json()["data"]["content_id"]

        response = client.delete(f"/api/v1/contents/{content_id}", headers=headers)
        assert response.status_code == 200

        detail_resp = client.get(f"/api/v1/contents/{content_id}", headers=headers)
        assert detail_resp.json()["data"]["status"] == "archived"
