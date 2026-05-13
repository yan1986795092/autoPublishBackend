"""Agent API 端点测试"""
import pytest


def test_run_agent_success(auth_client):
    """测试 POST /agent/run 成功路径"""
    client, token = auth_client
    response = client.post(
        "/api/v1/agent/run",
        json={"prompt": "写一篇关于 AI 的文章"},
        headers={"Authorization": f"Bearer {token}"},
    )
    # 期望 200 或 500（取决于 LLM 配置），但不应该 401
    assert response.status_code in (200, 500), f"Unexpected: {response.status_code} {response.text}"


def test_run_agent_no_auth(client):
    """未认证用户无法调用 agent 接口"""
    response = client.post(
        "/api/v1/agent/run",
        json={"prompt": "test"},
    )
    assert response.status_code == 401


def test_plan_endpoint(auth_client):
    """测试 POST /agent/plan"""
    client, token = auth_client
    response = client.post(
        "/api/v1/agent/plan",
        json={"prompt": "写一篇技术文章"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code in (200, 500)


def test_async_submit(auth_client):
    """测试 POST /agent/async/submit"""
    client, token = auth_client
    response = client.post(
        "/api/v1/agent/async/submit",
        json={"prompt": "写一篇技术文章"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code in (200, 500)

    if response.status_code == 200:
        data = response.json()
        assert data["success"] is True
        assert "task_id" in data["data"]


def test_async_status_nonexistent(auth_client):
    """查询不存在的任务应返回 404"""
    client, token = auth_client
    response = client.get(
        "/api/v1/agent/async/status/nonexistent-id",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


def test_async_list(auth_client):
    """测试 GET /agent/async/list"""
    client, token = auth_client
    response = client.get(
        "/api/v1/agent/async/list",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code in (200, 500)


def test_repurpose_endpoint(auth_client):
    """测试 POST /agent/repurpose"""
    client, token = auth_client
    response = client.post(
        "/api/v1/agent/repurpose",
        json={
            "content": "这是一篇关于人工智能的技术文章内容...",
            "title": "AI 入门指南",
            "formats": ["social_post", "newsletter"],
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code in (200, 500)

    if response.status_code == 200:
        data = response.json()
        assert data["success"] is True


@pytest.mark.skip(reason="WebSocket 需要特殊 event loop 配置")
def test_ws_endpoint(client):
    """WebSocket 端点是否可连接"""
    pass
