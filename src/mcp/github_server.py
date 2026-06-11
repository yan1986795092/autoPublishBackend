"""GitHub MCP Server — 代码与开发者内容搜索

基于 Model Context Protocol (MCP) SDK，
提供 GitHub API 的代码搜索、Issues、PRs 和仓库信息查询。

GitHub 特点：
- 官方 GitHub MCP Server (github/mcp-server)
- 代码搜索、Issues、PRs、仓库信息
- 适合从开源项目中提取内容、技术参考、案例研究
- Token 获取：https://github.com/settings/tokens
  免费层级：无需 Token 即可访问公开仓库（限速）
  推荐：创建 Classic Token (public_repo scope) 获得更高限额

如未配置 GITHUB_TOKEN，仍可访问公开仓库（更严格的限速）。
"""

from __future__ import annotations

import os
import logging

# 抑制 MCP SDK 内部 INFO 日志
logging.getLogger("mcp").setLevel(logging.WARNING)

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="github",
    instructions="GitHub — 代码与开发者内容搜索。搜索开源代码、Issues、PRs 和仓库信息。适用于技术文章写作、提取代码示例、案例研究等场景。",
    dependencies=["httpx>=0.25.0"],
)

_GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


@mcp.tool(
    name="search_code",
    description="在 GitHub 仓库中搜索代码。输入关键词，返回相关代码片段和所在仓库。适用于寻找代码示例、技术实现参考、开源库用法等。",
)
async def search_code(
    query: str,
    language: str = "",
    repo: str = "",
    max_results: int = 5,
) -> str:
    token = os.getenv("GITHUB_TOKEN")

    try:
        import httpx

        params = {
            "q": query,
            "per_page": min(max_results, 10),
            "sort": "indexed",
            "order": "desc",
        }
        if language:
            params["q"] += f" language:{language}"
        if repo:
            params["q"] += f" repo:{repo}"

        headers = dict(_GITHUB_HEADERS)
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.github.com/search/code",
                params=params,
                headers=headers,
            )

        if resp.status_code == 401:
            return "GitHub Token 无效或已过期。请检查 GITHUB_TOKEN 配置或创建新 Token：https://github.com/settings/tokens"
        if resp.status_code == 403:
            return "GitHub API 限速。请稍后再试或配置 GITHUB_TOKEN：https://github.com/settings/tokens"
        if resp.status_code != 200:
            return f"GitHub Code Search 请求失败（HTTP {resp.status_code}）：{resp.text}"

        data = resp.json()
        items = data.get("items", [])
        total = data.get("total_count", 0)

        if not items:
            return f"未找到与「{query}」相关的代码。尝试使用更通用的关键词。"

        lines = [f"**「{query}」GitHub 代码搜索结果（共 {total} 个，展示了 {len(items)} 个）：**\n"]
        for i, item in enumerate(items, 1):
            name = item.get("name", "")
            path = item.get("path", "")
            repo_full = item.get("repository", {}).get("full_name", "")
            html_url = item.get("html_url", "")
            score = item.get("score", 0)
            file_url = item.get("url", "")

            lines.append(f"**{i}. [{name}]({html_url})**")
            lines.append(f"   📁 路径：`{path}`")
            lines.append(f"   📦 仓库：[{repo_full}](https://github.com/{repo_full})")
            lines.append(f"   🔍 相关度：{score:.1f}")
            lines.append("")

        lines.append("_\n💡 代码来自 GitHub 开源项目，使用时请遵守各仓库的开源协议_")
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"GitHub code search error: {e}")
        return f"GitHub 代码搜索暂时不可用：{e}。请检查网络连接或配置 GITHUB_TOKEN。"


@mcp.tool(
    name="search_repositories",
    description="搜索 GitHub 仓库。按关键词查找相关仓库，返回星标数、描述、语言等信息。适用于寻找特定领域的开源项目、技术栈参考等。",
)
async def search_repositories(
    query: str,
    language: str = "",
    sort: str = "stars",
    max_results: int = 5,
) -> str:
    token = os.getenv("GITHUB_TOKEN")

    try:
        import httpx

        params = {
            "q": query,
            "per_page": min(max_results, 10),
            "sort": sort,
            "order": "desc",
        }
        if language:
            params["q"] += f" language:{language}"

        headers = dict(_GITHUB_HEADERS)
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.github.com/search/repositories",
                params=params,
                headers=headers,
            )

        if resp.status_code == 403:
            return "GitHub API 限速。稍后再试或配置 GITHUB_TOKEN。"
        if resp.status_code != 200:
            return f"GitHub 仓库搜索失败（HTTP {resp.status_code}）"

        data = resp.json()
        repos = data.get("items", [])
        total = data.get("total_count", 0)

        if not repos:
            return f"未找到与「{query}」相关的仓库。"

        lines = [f"**「{query}」热门仓库（共 {total} 个，展示 {len(repos)} 个）：**\n"]
        for i, repo in enumerate(repos, 1):
            full_name = repo.get("full_name", "")
            description = repo.get("description", "无描述")
            stars = repo.get("stargazers_count", 0)
            forks = repo.get("forks_count", 0)
            language = repo.get("language", "")
            html_url = repo.get("html_url", "")
            topics = repo.get("topics", [])[:5]

            lines.append(f"**{i}. [{full_name}]({html_url})** ⭐ {stars:,} | 🍴 {forks:,}")
            if description:
                lines.append(f"   {description[:120]}")
            if language:
                lines.append(f"   🏷️ 语言：{language}")
            if topics:
                lines.append(f"   📌 主题：{' '.join(f'`{t}`' for t in topics)}")
            lines.append("")

        lines.append("_\n💡 仓库来自 GitHub 开源项目，使用前请阅读 LICENSE 和 README_")
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"GitHub repo search error: {e}")
        return f"GitHub 仓库搜索暂时不可用：{e}"


@mcp.tool(
    name="get_issue",
    description="获取 GitHub Issue 详情。输入仓库和 Issue 编号，返回标题、正文、评论等信息。适用于从真实项目中提取问题背景、技术讨论等作为内容素材。",
)
async def get_issue(
    owner: str,
    repo: str,
    issue_number: int,
) -> str:
    token = os.getenv("GITHUB_TOKEN")

    if not owner or not repo:
        return "请提供 owner 和 repo 参数。格式：owner/repo（如：facebook/react）"

    try:
        import httpx

        headers = dict(_GITHUB_HEADERS)
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}",
                headers=headers,
            )

        if resp.status_code == 404:
            return f"未找到 {owner}/{repo} 的 Issue #{issue_number}。"
        if resp.status_code != 200:
            return f"获取 Issue 失败（HTTP {resp.status_code}）"

        issue = resp.json()
        title = issue.get("title", "")
        body = issue.get("body", "")[:1000]
        state = issue.get("state", "")
        labels = [l.get("name", "") for l in issue.get("labels", [])[:5]]
        author = issue.get("user", {}).get("login", "")
        created = issue.get("created_at", "")[:10]
        comments = issue.get("comments", 0)
        html_url = issue.get("html_url", "")

        lines = [
            f"**Issue #{issue_number}: {title}**",
            f"📦 [{owner}/{repo}](https://github.com/{owner}/{repo})",
            f"👤 作者：[{author}](https://github.com/{author}) | 📅 创建：{created} | 💬 评论：{comments}",
        ]
        if labels:
            lines.append(f"🏷️ 标签：{' '.join(f'`{l}`' for l in labels)}")
        lines.append(f"🔗 {html_url}\n")
        if body:
            lines.append(f"**正文：**\n{body}")
        else:
            lines.append("（无正文内容）")

        return "\n".join(lines)

    except Exception as e:
        return f"获取 Issue 失败：{e}"


@mcp.tool(
    name="readme_summary",
    description="获取 GitHub 仓库的 README 摘要。输入仓库名，返回 README 内容和关键信息提取。适用于快速了解一个开源项目的功能、用法、安装方法等。",
)
async def readme_summary(
    owner: str,
    repo: str,
    max_length: int = 2000,
) -> str:
    token = os.getenv("GITHUB_TOKEN")

    if not owner or not repo:
        return "请提供 owner 和 repo 参数。格式：owner/repo（如：microsoft/vscode）"

    try:
        import httpx

        headers = dict(_GITHUB_HEADERS)
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            # 获取 README
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/readme",
                headers=headers,
            )

        if resp.status_code == 404:
            return f"未找到 {owner}/{repo} 的 README。仓库可能为私有或不存在。"
        if resp.status_code != 200:
            return f"获取 README 失败（HTTP {resp.status_code}）"

        import base64
        data = resp.json()
        content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
        content = content[:max_length]
        if len(content) >= max_length:
            content += f"\n\n_(README 已被截断至 {max_length} 字)_"

        # 同时获取仓库基本信息
        async with httpx.AsyncClient(timeout=10.0) as client2:
            info_resp = await client2.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                headers=headers,
            )

        info_lines = [f"**📦 [{owner}/{repo}](https://github.com/{owner}/{repo}) README 摘要：**\n"]
        if info_resp.status_code == 200:
            info = info_resp.json()
            stars = info.get("stargazers_count", 0)
            forks = info.get("forks_count", 0)
            lang = info.get("language", "")
            desc = info.get("description", "")
            if desc:
                info_lines.append(f"📝 {desc}")
            info_lines.append(f"⭐ {stars:,} | 🍴 {forks:,}" + (f" | 🏷️ {lang}" if lang else ""))
            info_lines.append("")

        info_lines.append("---\n")
        info_lines.append(content)

        return "\n".join(info_lines)

    except Exception as e:
        return f"获取 README 失败：{e}"


if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "http":
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8770)
    else:
        mcp.run(transport="stdio")
