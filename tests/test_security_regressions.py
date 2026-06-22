import uuid
from datetime import datetime, timedelta, timezone

from src.agent.workflow.task_queue import AsyncTask, task_queue
from src.models import ChatConversation, Content, PublishRecord, PublishStatus


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_conversation_chat_blocks_cross_user_conversation(client, two_users, db_session):
    conversation = ChatConversation(user_id=two_users["user1"].id, title="secret")
    db_session.add(conversation)
    db_session.commit()
    db_session.refresh(conversation)

    response = client.post(
        "/api/v1/conversation/chat",
        json={"message": "hello", "conversation_id": conversation.id},
        headers=_auth_header(two_users["token2"]),
    )
    assert response.status_code == 404


def test_publish_accepts_timezone_aware_schedule(client, two_users, db_session):
    content = Content(
        user_id=two_users["user1"].id,
        title="hello",
        topic="topic",
        content_md="body",
    )
    db_session.add(content)
    db_session.commit()
    db_session.refresh(content)

    future_time = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    response = client.post(
        "/api/v1/publish",
        json={
            "content_id": content.id,
            "platform_code": "xiaohongshu",
            "publish_mode": "draft",
            "schedule_time": future_time,
        },
        headers=_auth_header(two_users["token1"]),
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["status"] == "scheduled"


def test_admin_metrics_require_auth(client):
    assert client.get("/api/v1/admin/metrics/production").status_code == 401
    assert client.get("/api/v1/admin/metrics/publishing").status_code == 401
    assert client.get("/api/v1/admin/metrics/cost").status_code == 401


def test_admin_metrics_are_user_scoped(client, two_users, db_session):
    own_content = Content(
        user_id=two_users["user1"].id,
        title="mine",
        topic="t1",
        platform_code="wechat",
        content_md="body",
        metadata_json='{"ai": true}',
    )
    other_content = Content(
        user_id=two_users["user2"].id,
        title="other",
        topic="t2",
        platform_code="zhihu",
        content_md="body",
    )
    db_session.add_all([own_content, other_content])
    db_session.commit()
    db_session.refresh(own_content)
    db_session.refresh(other_content)

    db_session.add_all(
        [
            PublishRecord(content_id=own_content.id, platform_code="wechat", status=PublishStatus.SUCCESS),
            PublishRecord(content_id=other_content.id, platform_code="zhihu", status=PublishStatus.FAILED),
        ]
    )
    db_session.commit()

    production = client.get("/api/v1/admin/metrics/production", headers=_auth_header(two_users["token1"]))
    publishing = client.get("/api/v1/admin/metrics/publishing", headers=_auth_header(two_users["token1"]))
    cost = client.get("/api/v1/admin/metrics/cost", headers=_auth_header(two_users["token1"]))

    assert production.status_code == 200
    assert publishing.status_code == 200
    assert cost.status_code == 200
    platform_distribution = production.json()["data"]["platform_distribution"]
    assert all(item["platform_code"] != "zhihu" for item in platform_distribution)
    assert production.json()["data"]["ai_generation_ratio"] >= 0.0
    publishing_data = publishing.json()["data"]
    assert all(item["platform_code"] != "zhihu" for item in publishing_data["platforms"])
    assert all(item["platform_code"] != "zhihu" for item in publishing_data["failures"])
    assert cost.json()["data"]["monthly"]["total_cost"] >= 0


def test_private_urls_are_rejected(client, auth_client):
    client, token = auth_client

    skill_resp = client.post(
        "/api/v1/skills/import/url",
        json={"url": "http://127.0.0.1/internal"},
        headers=_auth_header(token),
    )
    assert skill_resp.status_code == 400

    parse_resp = client.post(
        "/api/v1/contents/ai-edit/parse-url",
        json={"url": "http://localhost:3000"},
        headers=_auth_header(token),
    )
    assert parse_resp.status_code == 200
    assert parse_resp.json()["success"] is False

    crawl_resp = client.post(
        "/api/v1/agent/crawl",
        json={"url": "http://10.0.0.1"},
        headers=_auth_header(token),
    )
    assert crawl_resp.status_code == 400


def test_websocket_requires_task_owner(client, two_users):
    task = AsyncTask(id=str(uuid.uuid4()), user_id=two_users["user1"].id, prompt="ws")
    task_queue._tasks[task.id] = task

    with client.websocket_connect(f"/api/v1/agent/ws/{task.id}?token={two_users['token1']}") as websocket:
        message = websocket.receive_json()
        assert message["type"] == "progress"

    try:
        client.websocket_connect(f"/api/v1/agent/ws/{task.id}?token={two_users['token2']}")
        assert False, "expected websocket auth failure"
    except Exception:
        pass
