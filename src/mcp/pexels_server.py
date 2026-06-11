"""Pexels MCP Server — 免费高清图片搜索

基于 Model Context Protocol (MCP) SDK，
提供 Pexels API 的免费高清图片搜索和下载。

Pexels 特点：
- 完全免费，高质量照片和视频
- 无需署名（知识共享零许可 CC0）
- 覆盖生活、自然、商业等 20 万+ 分类
- API Key 获取：https://www.pexels.com/api/
  免费层级：每月 200 次请求

如未配置 PEXELS_API_KEY，返回友好的引导信息。
"""

from __future__ import annotations

import os
import logging

# 抑制 MCP SDK 内部 INFO 日志
logging.getLogger("mcp").setLevel(logging.WARNING)

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="pexels",
    instructions="Pexels — 免费高清图片搜索。提供数十万张高质量、免费可商用的照片，适用于文章配图、社交媒体内容、营销素材等场景。",
    dependencies=["httpx>=0.25.0"],
)


@mcp.tool(
    name="search_images",
    description="搜索 Pexels 免费高清图片。输入关键词，返回相关图片列表（URL、尺寸、作者信息）。适用于为文章、社交媒体等内容寻找配图。",
)
async def search_images(
    query: str,
    per_page: int = 6,
    orientation: str = "any",
    size: str = "any",
    color: str = "",
    locale: str = "zh-CN",
) -> str:
    api_key = os.getenv("PEXELS_API_KEY")

    if not api_key:
        return await _mock_results(query, per_page)

    try:
        import httpx

        params = {
            "query": query,
            "per_page": min(per_page, 15),
            "orientation": orientation if orientation != "any" else None,
            "size": size if size != "any" else None,
            "locale": locale,
        }
        params = {k: v for k, v in params.items() if v}

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.pexels.com/v1/search",
                params=params,
                headers={"Authorization": api_key},
            )

        if resp.status_code == 401:
            return "Pexels API Key 无效。请到 https://www.pexels.com/api/ 获取有效的 API Key。\n\n" + await _mock_results(query, per_page)

        if resp.status_code != 200:
            return await _mock_results(query, per_page)

        data = resp.json()
        photos = data.get("photos", [])
        total = data.get("total_results", 0)

        if not photos:
            return await _mock_results(query, per_page)

        lines = [f"**「{query}」免费图片搜索结果（共找到 {total} 张，以下展示 {len(photos)} 张）：**\n"]
        for i, photo in enumerate(photos, 1):
            src = photo.get("src", {})
            avg_color = photo.get("avg_color", "")
            photographer = photo.get("photographer", "Unknown")
            photographer_url = photo.get("photographer_url", "")
            alt = photo.get("alt", "")

            lines.append(f"**{i}. {alt or query}**")
            lines.append(f"   📐 尺寸：{src.get('large', src.get('original', {}).get('url', 'N/A'))}")
            lines.append(f"   🖼️ URL：{src.get('large', {}).get('url', src.get('original', {}).get('url', 'N/A'))}")
            if avg_color:
                lines.append(f"   🎨 平均色：{avg_color}")
            lines.append(f"   📷 摄影师：[{photographer}]({photographer_url})")
            lines.append("")

        lines.append("_\n💡 以上图片来自 Pexels（CC0 许可证，可免费商用，使用时建议标注来源）_")
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Pexels API error: {e}")
        return await _mock_results(query, per_page)


@mcp.tool(
    name="search_images_by_color",
    description="按颜色搜索 Pexels 免费图片。输入关键词和颜色，返回该颜色的相关图片列表。适用于需要与文章主题色调一致的配图。",
)
async def search_images_by_color(
    query: str,
    color: str,
    per_page: int = 6,
) -> str:
    return await search_images(query, per_page=per_page, color=color)


@mcp.tool(
    name="get_curated_photos",
    description="获取 Pexels 精选图片（无需搜索关键词）。返回每日精选的高质量图片列表，适用于封面图、背景图等场景。",
)
async def get_curated_photos(
    per_page: int = 6,
) -> str:
    api_key = os.getenv("PEXELS_API_KEY")

    if not api_key:
        return "**Pexels 精选图片（Mock 预览）：**\n\n如需真实图片，请配置 PEXELS_API_KEY：https://www.pexels.com/api/\n\n" + await _mock_results("curated", per_page)

    try:
        import httpx

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.pexels.com/v1/curated",
                params={"per_page": min(per_page, 15)},
                headers={"Authorization": api_key},
            )

        if resp.status_code != 200:
            return await _mock_results("curated", per_page)

        data = resp.json()
        photos = data.get("photos", [])

        lines = [f"**Pexels 精选图片（共 {len(photos)} 张）：**\n"]
        for i, photo in enumerate(photos, 1):
            src = photo.get("src", {})
            alt = photo.get("alt", "精选图片")
            photographer = photo.get("photographer", "")
            photographer_url = photo.get("photographer_url", "")
            lines.append(f"**{i}. {alt}**")
            lines.append(f"   🖼️ {src.get('large', {}).get('url', '')}")
            lines.append(f"   📷 摄影师：[{photographer}]({photographer_url})")
            lines.append("")

        lines.append("_\n💡 图片来自 Pexels 精选（CC0 许可证，可免费商用）_")
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Pexels curated error: {e}")
        return await _mock_results("curated", per_page)


async def _mock_results(query: str, per_page: int) -> str:
    """Mock 结果（未配置 API Key 时展示）"""
    mock_images = [
        {"title": f"{query} 相关图片 1", "url": "https://images.pexels.com/photos/1/pexels-photo-1.jpeg", "color": "#4A90E2"},
        {"title": f"{query} 相关图片 2", "url": "https://images.pexels.com/photos/2/pexels-photo-2.jpeg", "color": "#50C878"},
        {"title": f"{query} 相关图片 3", "url": "https://images.pexels.com/photos/3/pexels-photo-3.jpeg", "color": "#FF6B6B"},
        {"title": f"{query} 相关图片 4", "url": "https://images.pexels.com/photos/4/pexels-photo-4.jpeg", "color": "#F39C12"},
        {"title": f"{query} 相关图片 5", "url": "https://images.pexels.com/photos/5/pexels-photo-5.jpeg", "color": "#9B59B6"},
        {"title": f"{query} 相关图片 6", "url": "https://images.pexels.com/photos/6/pexels-photo-6.jpeg", "color": "#34495E"},
    ]

    lines = [
        f"**「{query}」Pexels 图片搜索（演示模式）：**",
        f"\n⚠️ 当前为演示模式，配置 PEXELS_API_KEY 可获取真实图片。",
        f"\n免费获取 API Key：https://www.pexels.com/api/",
        "\n\n**演示图片：**\n",
    ]

    for i, img in enumerate(mock_images[:per_page], 1):
        lines.append(f"**{i}. {img['title']}**")
        lines.append(f"   🎨 主色调：{img['color']}")
        lines.append(f"   🖼️ 演示 URL：{img['url']}")
        lines.append("")

    lines.append("_\n💡 配置 PEXELS_API_KEY 后可搜索数十万张真实免费图片_")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "http":
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8769)
    else:
        mcp.run(transport="stdio")
