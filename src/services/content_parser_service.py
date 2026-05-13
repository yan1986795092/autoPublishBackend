import re
from urllib.parse import urlparse
import httpx


class ContentParserService:
    """URL 内容提取服务（MVP）"""

    async def parse_url(self, url: str) -> dict:
        """从 URL 提取标题、正文和图片"""
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {"success": False, "error": "无效的 URL 格式"}

        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                })
                resp.raise_for_status()
                html = resp.text
        except httpx.TimeoutException:
            return {"success": False, "error": "请求超时"}
        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

        # 提取标题
        title = ""
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = self._clean_text(title_match.group(1))

        # 提取 meta description
        description = ""
        desc_match = re.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']*)["\']',
            html, re.IGNORECASE
        )
        if not desc_match:
            desc_match = re.search(
                r'<meta[^>]+content=["\']([^"\']*)["\'][^>]+name=["\']description["\']',
                html, re.IGNORECASE
            )
        if desc_match:
            description = self._clean_text(desc_match.group(1))

        # 提取正文（去除 HTML 标签）
        body_text = self._extract_body_text(html)
        body_text = body_text[:5000]  # 截断避免过长

        # 提取图片
        images = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE)
        # 过滤 data: URI 和小图标
        valid_images = [
            img for img in images[:10]
            if not img.startswith("data:") and not img.endswith((".svg", ".ico"))
        ]

        return {
            "success": True,
            "data": {
                "title": title,
                "description": description,
                "content": body_text,
                "images": valid_images,
                "url": url,
                "domain": parsed.netloc,
            }
        }

    def _extract_body_text(self, html: str) -> str:
        """提取可读正文（移除 script/style 标签后提取文本）"""
        # 移除 script, style, nav, footer 等
        for tag in ['script', 'style', 'nav', 'footer', 'header', 'noscript']:
            html = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', '', html, flags=re.IGNORECASE | re.DOTALL)

        # 移除所有 HTML 标签
        text = re.sub(r'<[^>]+>', ' ', html)
        # 合并空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        # 按句分割，过滤空行和过短行
        lines = [line.strip() for line in re.split(r'[。！？\n]', text) if len(line.strip()) > 10]
        return "。".join(lines[:100])

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:500]


content_parser_service = ContentParserService()
