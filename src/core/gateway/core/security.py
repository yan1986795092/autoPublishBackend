from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from typing import Optional
import hashlib


class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        allowed_source_apps: list[str],
        allowed_models: list[str],
        max_content_length: int = 32768,
    ):
        super().__init__(app)
        self.allowed_source_apps = set(allowed_source_apps)
        self.allowed_models = set(allowed_models)
        self.max_content_length = max_content_length

    async def dispatch(self, request: Request, call_next):
        source_app = request.headers.get("X-Source-App", "default")
        if self.allowed_source_apps and source_app not in self.allowed_source_apps:
            return JSONResponse(
                status_code=403,
                content={"error": {"code": "forbidden", "message": f"Source app '{source_app}' not allowed"}},
                headers={"X-Source-App": source_app},
            )
        response = await call_next(request)
        response.headers["X-Source-App"] = source_app
        return response


def validate_request_body(body: dict, allowed_models: set, max_content_length: int) -> Optional[str]:
    model = body.get("model", "")
    if not model:
        return "model field is required"

    if allowed_models and model not in allowed_models:
        return f"Model '{model}' is not in the allowed list"

    messages = body.get("messages", [])
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > max_content_length:
            return f"Message content exceeds {max_content_length} characters"

    return None


def mask_message(content: str) -> str:
    if not content:
        return ""
    short = content[:20]
    hash_val = hashlib.sha256(content.encode()).hexdigest()[:8]
    return f"{short}...[hash:{hash_val}]"
