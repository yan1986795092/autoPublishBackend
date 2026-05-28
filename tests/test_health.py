"""Tests for the enhanced /health endpoint."""

import pytest


class TestHealthEndpoint:
    """GET /health 端点测试"""

    def test_health_returns_healthy(self, client):
        """基础健康检查返回 healthy"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")
        assert "timestamp" in data

    def test_health_has_checks(self, client):
        """health 检查包含 subsystems"""
        response = client.get("/health")
        data = response.json()
        assert "checks" in data
        assert "database" in data["checks"]
        assert "worker" in data["checks"]

    def test_health_db_check_value(self, client):
        """database 检查返回 passed 或 failed"""
        response = client.get("/health")
        data = response.json()
        assert data["checks"]["database"] in ("passed", "failed")

    def test_health_worker_check_value(self, client):
        """worker 检查返回 running 或 stopped"""
        response = client.get("/health")
        data = response.json()
        assert data["checks"]["worker"] in ("running", "stopped")

    def test_health_different_from_root(self, client):
        """/health 返回结构不同于 /"""
        health = client.get("/health").json()
        root = client.get("/").json()
        assert "checks" in health
        assert "checks" not in root
