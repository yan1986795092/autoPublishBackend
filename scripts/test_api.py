#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoPublish API Test Script
Usage: python scripts/test_api.py [--base-url http://localhost:8000]

Tests all core API endpoints in sequence:
1. Register a test user
2. Login
3. Get user info
4. List platforms
5. List templates
6. Generate content
7. List contents
8. Get content detail
9. Update content
10. Publish content
11. Delete content
"""

import argparse
import json
import time
import sys
import io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import requests
except ImportError:
    print("[X] 'requests' library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests


class APITester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.token = None
        self.refresh_token = None
        self.user_id = None
        self.content_id = None
        self.email = None
        self.passed = 0
        self.failed = 0
        self.start_time = time.time()

    def headers(self, auth: bool = True) -> dict:
        h = {"Content-Type": "application/json"}
        if auth and self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def request(self, method: str, path: str, data: dict = None, auth: bool = True, desc: str = "") -> requests.Response:
        url = f"{self.base_url}{path}"
        try:
            if method.upper() == "GET":
                r = requests.get(url, headers=self.headers(auth), params=data or {})
            elif method.upper() == "POST":
                r = requests.post(url, headers=self.headers(auth), json=data or {})
            elif method.upper() == "PUT":
                r = requests.put(url, headers=self.headers(auth), json=data or {})
            elif method.upper() == "DELETE":
                r = requests.delete(url, headers=self.headers(auth))
            else:
                raise ValueError(f"Unsupported method: {method}")

            status = "[OK]" if r.status_code < 400 else "[FAIL]"
            elapsed = f"{r.elapsed.total_seconds()*1000:.0f}ms"
            try:
                body = r.json()
                success = body.get("success", False)
            except:
                body = {"raw": r.text[:200]}
                success = r.status_code < 400

            print(f"  {status} [{method}] {path} → {r.status_code} ({elapsed})")
            if desc:
                print(f"     {desc}")

            if r.status_code >= 400:
                try:
                    err = r.json()
                    print(f"     Error: {err}")
                except:
                    pass

            return r
        except requests.exceptions.ConnectionError:
            print(f"  [X] [CONNECTION ERROR] {path} - Is the server running?")
            sys.exit(1)
        except Exception as e:
            print(f"  [X] [ERROR] {path} - {e}")
            return None

    def section(self, title: str):
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    def ok(self, msg: str = ""):
        self.passed += 1
        if msg:
            print(f"     [OK] {msg}")

    def fail(self, msg: str):
        self.failed += 1
        print(f"     [FAIL] {msg}")

    def test_auth_register(self) -> bool:
        self.section("1. 用户注册 (POST /api/v1/auth/register)")

        # Generate unique email using a valid domain
        ts = int(time.time() * 1000)
        self.email = f"test_{ts}@example.com"

        data = {
            "email": self.email,
            "password": "TestPass123!",
            "nickname": "测试用户"
        }
        r = self.request("POST", "/api/v1/auth/register", data, auth=False)
        if not r:
            return False

        body = r.json()
        if body.get("success"):
            self.user_id = body.get("data", {}).get("user_id")
            self.ok(f"注册成功: {self.email}")
            return True
        else:
            # If already exists, that's okay for testing
            if "already" in str(body).lower():
                self.ok(f"用户已存在（跳过注册）")
                return True
            self.fail(f"注册失败: {body}")
            return False

    def test_auth_login(self) -> bool:
        self.section("2. 用户登录 (POST /api/v1/auth/login)")

        data = {
            "email": self.email or "test@example.com",
            "password": "TestPass123!"
        }
        r = self.request("POST", "/api/v1/auth/login", data, auth=False)
        if not r:
            return False

        body = r.json()
        if body.get("success") and body.get("data"):
            self.token = body["data"].get("access_token")
            self.refresh_token = body["data"].get("refresh_token")
            self.ok(f"登录成功, Token: {self.token[:20]}...")
            return True
        else:
            self.fail(f"登录失败: {body}")
            return False

    def test_auth_me(self) -> bool:
        self.section("3. 获取用户信息 (GET /api/v1/auth/me)")

        r = self.request("GET", "/api/v1/auth/me")
        if not r:
            return False

        body = r.json()
        if body.get("success"):
            user = body.get("data", {})
            self.ok(f"用户: {user.get('email')} (配额: {user.get('quota_daily')} / 已用: {user.get('quota_used_today')})")
            return True
        else:
            self.fail(f"获取用户信息失败: {body}")
            return False

    def test_platforms(self) -> bool:
        self.section("4. 平台列表 (GET /api/v1/platforms)")

        r = self.request("GET", "/api/v1/platforms")
        if not r:
            return False

        body = r.json()
        if body.get("success"):
            platforms = body.get("data", [])
            for p in platforms:
                self.ok(f"平台: {p.get('name')} ({p.get('code')}) - Mock: {p.get('is_mock')}")
            return True
        else:
            self.fail(f"获取平台列表失败: {body}")
            return False

    def test_templates(self) -> bool:
        self.section("5. 模板列表 (GET /api/v1/templates)")

        r = self.request("GET", "/api/v1/templates")
        if not r:
            return False

        body = r.json()
        if body.get("success"):
            templates = body.get("data", [])
            for t in templates:
                self.ok(f"模板: {t.get('name')} (使用 {t.get('use_count')} 次)")
            return True
        else:
            self.fail(f"获取模板列表失败: {body}")
            return False

    def test_content_generate(self) -> bool:
        self.section("6. AI 内容生成 (POST /api/v1/contents/generate)")

        data = {
            "topic": "职场沟通技巧：如何优雅地拒绝同事的不合理请求",
            "platform_code": "xiaohongshu",
            "parameters": {
                "length": "medium",
                "creativity": 0.7,
                "tone": "friendly"
            }
        }
        r = self.request("POST", "/api/v1/contents/generate", data)
        if not r:
            return False

        body = r.json()
        if body.get("success"):
            content_id = body.get("data", {}).get("content_id")
            self.content_id = content_id
            meta = body.get("data", {}).get("metadata", {})
            self.ok(f"生成成功: ID={content_id}")
            self.ok(f"模型: {meta.get('model')}, Token: {meta.get('tokens_used')}, 耗时: {meta.get('generation_time_ms')}ms")
            title = body.get("data", {}).get("title", "")
            content_preview = body.get("data", {}).get("content_md", "")[:100]
            self.ok(f"标题: {title}")
            self.ok(f"内容预览: {content_preview}...")
            return True
        else:
            self.fail(f"内容生成失败: {body}")
            return False

    def test_content_list(self) -> bool:
        self.section("7. 内容列表 (GET /api/v1/contents)")

        r = self.request("GET", "/api/v1/contents", {"page": 1, "page_size": 10})
        if not r:
            return False

        body = r.json()
        if body.get("success"):
            items = body.get("data", {}).get("items", [])
            pagination = body.get("data", {}).get("pagination", {})
            self.ok(f"共 {pagination.get('total', 0)} 条内容, 第 {pagination.get('page')}/{pagination.get('total_pages')} 页")
            for c in items:
                self.ok(f"  - {c.get('title', '无标题')} [{c.get('status')}]")
            return True
        else:
            self.fail(f"获取内容列表失败: {body}")
            return False

    def test_content_get(self) -> bool:
        self.section("8. 内容详情 (GET /api/v1/contents/{id})")

        if not self.content_id:
            self.fail("无 content_id，跳过")
            return False

        r = self.request("GET", f"/api/v1/contents/{self.content_id}")
        if not r:
            return False

        body = r.json()
        if body.get("success"):
            data = body.get("data", {})
            self.ok(f"标题: {data.get('title')}")
            self.ok(f"平台: {data.get('platform_code')}")
            self.ok(f"状态: {data.get('status')}")
            versions = data.get("versions", [])
            self.ok(f"版本数: {len(versions)}")
            return True
        else:
            self.fail(f"获取内容详情失败: {body}")
            return False

    def test_content_update(self) -> bool:
        self.section("9. 更新内容 (PUT /api/v1/contents/{id})")

        if not self.content_id:
            self.fail("无 content_id，跳过")
            return False

        data = {
            "title": f"测试标题 - {datetime.now().strftime('%H:%M:%S')}",
            "content_md": "# 测试更新\n\n这是一段测试更新的内容。\n\n- 第一点\n- 第二点\n- 第三点",
        }
        r = self.request("PUT", f"/api/v1/contents/{self.content_id}", data)
        if not r:
            return False

        body = r.json()
        if body.get("success"):
            self.ok("更新成功")
            return True
        else:
            self.fail(f"更新失败: {body}")
            return False

    def test_publish(self) -> bool:
        self.section("10. 发布内容 (POST /api/v1/publish)")

        if not self.content_id:
            self.fail("无 content_id，跳过")
            return False

        data = {
            "content_id": self.content_id,
            "platform_code": "xiaohongshu",
            "publish_mode": "draft"
        }
        r = self.request("POST", "/api/v1/publish", data)
        if not r:
            return False

        body = r.json()
        if body.get("success"):
            result = body.get("data", {})
            self.ok(f"发布状态: {result.get('status')}")
            self.ok(f"平台: {result.get('platform')}")
            if result.get("platform_post_id"):
                self.ok(f"平台帖子ID: {result.get('platform_post_id')}")
            return True
        else:
            self.fail(f"发布失败: {body}")
            return False

    def test_content_delete(self) -> bool:
        self.section("11. 删除内容 (DELETE /api/v1/contents/{id})")

        if not self.content_id:
            self.fail("无 content_id，跳过")
            return False

        r = self.request("DELETE", f"/api/v1/contents/{self.content_id}")
        if not r:
            return False

        body = r.json()
        if body.get("success"):
            self.ok("删除成功")
            return True
        else:
            self.fail(f"删除失败: {body}")
            return False

    def run(self):
        print(f"\n{'#'*60}")
        print(f"#  AutoPublish API 测试脚本")
        print(f"#  Base URL: {self.base_url}")
        print(f"#  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}")

        steps = [
            (self.test_auth_register, "注册"),
            (self.test_auth_login, "登录"),
            (self.test_auth_me, "获取用户信息"),
            (self.test_platforms, "平台列表"),
            (self.test_templates, "模板列表"),
            (self.test_content_generate, "生成内容"),
            (self.test_content_list, "内容列表"),
            (self.test_content_get, "内容详情"),
            (self.test_content_update, "更新内容"),
            (self.test_publish, "发布内容"),
            (self.test_content_delete, "删除内容"),
        ]

        for step, name in steps:
            try:
                step()
            except Exception as e:
                self.fail(f"测试异常: {e}")
                import traceback
                traceback.print_exc()

        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"  测试结果汇总")
        print(f"{'='*60}")
        print(f"  [OK] 通过: {self.passed}")
        print(f"  [X] 失败: {self.failed}")
        print(f"  ⏱️  总耗时: {total_time:.1f}s")
        print(f"{'='*60}\n")

        if self.failed > 0:
            sys.exit(1)
        else:
            print("*** ALL TESTS PASSED! ***")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="AutoPublish API Test Script")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )
    args = parser.parse_args()

    tester = APITester(args.base_url)
    tester.run()


if __name__ == "__main__":
    main()
