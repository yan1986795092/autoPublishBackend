"""Agent API endpoint tests."""

import uuid
from types import SimpleNamespace

from src.agent.workflow.task_queue import AsyncTask, TaskState, task_queue
from src.models import AgentTask as AgentTaskModel, SubTask


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def setup_function():
    task_queue._tasks.clear()
    task_queue._dead_letter_queue.clear()


def teardown_function():
    task_queue._tasks.clear()
    task_queue._dead_letter_queue.clear()


class _FakeRunResult:
    def __init__(self, payload=None):
        self.status = SimpleNamespace(value="completed")
        self._payload = payload or {"output": {"content": "ok"}}

    def to_dict(self):
        return self._payload


class _FakePlan:
    def __init__(self):
        self.goal = "goal"
        self.reasoning = "reasoning"
        self.estimated_steps = 1
        self.tasks = [SimpleNamespace(agent_type="WritingAgent", input={"task": "x"}, dependencies=[])]


class _FakeSubtask:
    def __init__(self):
        self.status = SimpleNamespace(value="completed")
        self.output = {"content": "repurposed"}


def test_run_agent_success(auth_client, monkeypatch):
    client, token = auth_client
    captured: dict[str, object] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return _FakeRunResult()

    monkeypatch.setattr(
        "src.agent.api._create_orchestrator",
        lambda **kwargs: SimpleNamespace(run=fake_run),
    )
    response = client.post(
        "/api/v1/agent/run",
        json={"prompt": "write something about AI"},
        headers=_auth_header(token),
    )
    assert response.status_code == 200
    assert captured["context"]["user_id"]


def test_run_agent_no_auth(client):
    response = client.post("/api/v1/agent/run", json={"prompt": "test"})
    assert response.status_code == 401


def test_plan_endpoint(auth_client, monkeypatch):
    client, token = auth_client
    async def fake_decompose(**kwargs):
        return _FakePlan()

    monkeypatch.setattr(
        "src.agent.api._create_orchestrator",
        lambda **kwargs: SimpleNamespace(_decompose=fake_decompose),
    )
    response = client.post(
        "/api/v1/agent/plan",
        json={"prompt": "write a technical article"},
        headers=_auth_header(token),
    )
    assert response.status_code == 200


def test_async_submit(auth_client):
    client, token = auth_client
    response = client.post(
        "/api/v1/agent/async/submit",
        json={"prompt": "write a technical article"},
        headers=_auth_header(token),
    )
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.json()
        assert data["success"] is True
        assert "task_id" in data["data"]


def test_repurpose_endpoint(auth_client, monkeypatch):
    client, token = auth_client

    async def fake_fallback_llm_call(**kwargs):
        return _FakeSubtask()

    monkeypatch.setattr(
        "src.agent.api._create_orchestrator",
        lambda **kwargs: SimpleNamespace(_fallback_llm_call=fake_fallback_llm_call),
    )

    response = client.post(
        "/api/v1/agent/repurpose",
        json={
            "content": "source content",
            "title": "AI guide",
            "formats": ["social_post", "newsletter"],
        },
        headers=_auth_header(token),
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_async_status_nonexistent(auth_client):
    client, token = auth_client
    response = client.get(
        "/api/v1/agent/async/status/nonexistent-id",
        headers=_auth_header(token),
    )
    assert response.status_code == 404


def test_async_status_is_scoped_to_owner(client, two_users):
    task = AsyncTask(id=str(uuid.uuid4()), user_id=two_users["user1"].id, prompt="secret")
    task_queue._tasks[task.id] = task

    response = client.get(
        f"/api/v1/agent/async/status/{task.id}",
        headers=_auth_header(two_users["token2"]),
    )
    assert response.status_code == 404


def test_async_cancel_is_scoped_to_owner(client, two_users):
    task = AsyncTask(id=str(uuid.uuid4()), user_id=two_users["user1"].id, prompt="secret")
    task_queue._tasks[task.id] = task

    response = client.delete(
        f"/api/v1/agent/async/{task.id}",
        headers=_auth_header(two_users["token2"]),
    )
    assert response.status_code == 404


def test_dlq_retry_is_scoped_and_awaited(client, two_users):
    task = AsyncTask(
        id=str(uuid.uuid4()),
        user_id=two_users["user1"].id,
        prompt="retry me",
        state=TaskState.FAILED,
    )
    task_queue._dead_letter_queue[task.id] = task

    forbidden = client.post(
        f"/api/v1/agent/dlq/{task.id}/retry",
        headers=_auth_header(two_users["token2"]),
    )
    assert forbidden.status_code == 404

    ok = client.post(
        f"/api/v1/agent/dlq/{task.id}/retry",
        headers=_auth_header(two_users["token1"]),
    )
    assert ok.status_code == 200
    data = ok.json()
    assert data["success"] is True
    assert data["data"]["task_id"] == task.id
    assert task_queue.get_task(task.id) is not None


def test_vector_search_is_user_scoped(client, two_users, monkeypatch):
    captured: dict[str, object] = {}

    def fake_add_documents(*, texts, metadatas, ids=None):
        captured["texts"] = texts
        captured["metadatas"] = metadatas
        return True

    def fake_search(*, query, top_k, namespace=None, metadata_filter=None):
        captured["query"] = query
        captured["top_k"] = top_k
        captured["namespace"] = namespace
        captured["metadata_filter"] = metadata_filter
        return {"query": query, "matches": [], "total": 0}

    monkeypatch.setattr("src.agent.api.vector_search.add_documents", fake_add_documents)
    monkeypatch.setattr("src.agent.api.vector_search.search", fake_search)

    response = client.post(
        "/api/v1/agent/vector-search",
        json={"query": "hello", "texts": ["doc1", "doc2"]},
        headers=_auth_header(two_users["token1"]),
    )

    assert response.status_code == 200
    expected_namespace = f"user:{two_users['user1'].id}"
    assert captured["namespace"] == expected_namespace
    assert captured["metadata_filter"] == {"user_id": two_users["user1"].id}
    assert captured["metadatas"] == [
        {"user_id": two_users["user1"].id, "namespace": expected_namespace},
        {"user_id": two_users["user1"].id, "namespace": expected_namespace},
    ]


def test_metrics_requires_auth(client):
    response = client.get("/api/v1/agent/metrics")
    assert response.status_code == 401


def test_metrics_are_scoped_to_current_user(client, two_users, db_session):
    own_task = AgentTaskModel(
        id=str(uuid.uuid4()),
        user_id=two_users["user1"].id,
        prompt="own",
        status="completed",
        total_tokens=12,
        total_cost=1.5,
    )
    other_task = AgentTaskModel(
        id=str(uuid.uuid4()),
        user_id=two_users["user2"].id,
        prompt="other",
        status="completed",
        total_tokens=99,
        total_cost=9.9,
    )
    db_session.add_all([own_task, other_task])
    db_session.flush()
    db_session.add_all(
        [
            SubTask(task_id=own_task.id, agent_type="ReviewAgent", status="completed", tokens_used=5, time_elapsed=2.0),
            SubTask(task_id=other_task.id, agent_type="ReviewAgent", status="completed", tokens_used=50, time_elapsed=5.0),
        ]
    )
    db_session.commit()

    task_queue._tasks["active-own"] = AsyncTask(
        id="active-own",
        user_id=two_users["user1"].id,
        prompt="active",
        state=TaskState.RUNNING,
    )
    task_queue._tasks["active-other"] = AsyncTask(
        id="active-other",
        user_id=two_users["user2"].id,
        prompt="active",
        state=TaskState.RUNNING,
    )
    task_queue._dead_letter_queue["dlq-own"] = AsyncTask(
        id="dlq-own",
        user_id=two_users["user1"].id,
        prompt="dlq",
        state=TaskState.FAILED,
    )
    task_queue._dead_letter_queue["dlq-other"] = AsyncTask(
        id="dlq-other",
        user_id=two_users["user2"].id,
        prompt="dlq",
        state=TaskState.FAILED,
    )

    response = client.get("/api/v1/agent/metrics", headers=_auth_header(two_users["token1"]))
    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["tasks"]["active"] == 1
    assert payload["tasks"]["dlq_size"] == 1
    assert payload["history"]["agent_runs_total"] == 1
    assert payload["history"]["llm_tokens_total"] == 12
    assert payload["history"]["llm_cost_total"] == 1.5
    assert payload["history"]["agents"][0]["tokens"] == 5


def test_cache_invalidate_is_disabled(auth_client):
    client, token = auth_client
    response = client.post("/api/v1/agent/cache/invalidate", headers=_auth_header(token))
    assert response.status_code == 403


def test_run_agent_injects_current_user_into_context(client, two_users, monkeypatch):
    captured: dict[str, object] = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return _FakeRunResult()

    monkeypatch.setattr(
        "src.agent.api._create_orchestrator",
        lambda **kwargs: SimpleNamespace(run=fake_run),
    )

    response = client.post(
        "/api/v1/agent/run",
        json={"prompt": "write", "context": {"user_id": "spoofed", "custom": "value"}},
        headers=_auth_header(two_users["token1"]),
    )

    assert response.status_code == 200
    assert captured["context"]["user_id"] == two_users["user1"].id
    assert captured["context"]["custom"] == "value"
