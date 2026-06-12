from fastapi import FastAPI, Request, WebSocketDisconnect, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import asyncio

from src.core.exceptions import (
    GenerateError, ServiceError, AgentError, HandlerError, ToolExecutionError,
    ConfigurationError, ValidationError, AuthenticationError, AuthorizationError,
    NotFoundError, ConflictError, RateLimitError, build_error_response,
)
from src.logging_shared.config import setup as setup_logging
from src.core.migrations import migration_manager
from src.config import get_settings
from src.database import engine, setup_db_logging
from src.models import Base
from src.api import auth, contents, publish, platforms, presets, ai_editor, personas
from src.api.admin import router as admin_router
from src.api.skills import router as skills_router
from src.api.chat import router as chat_router
from src.agent.api import router as agent_router
from src.agent.workflow.worker import worker
from src.observability import setup_observability
from src.observability.metrics import up_gauge, info_gauge
from src.logging_shared.unified_writer import UnifiedEsWriter
import src.migrations  # noqa: F401 — 注册迁移

import structlog
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # 统一日志配置（读取 settings.log_level, settings.log_json_format）
    setup_logging(settings)

    # 统一 tracing 初始化（structlog only，OTel SDK 已移除）
    tracer, meter = setup_observability(
        service_name="autopublish-api",
        service_version="2.0.0",
        settings=settings,
    )

    # 设置 Prometheus 生命周期仪表指标
    up_gauge.set(1)
    info_gauge.labels(version="2.0.0").set(1)

    # 开发环境启用 DB 查询日志（默认关，可通过 ENABLE_DB_LOG=true 开启）
    setup_db_logging(enabled=bool(getattr(settings, "log_level", "INFO") == "DEBUG"))

    # 启动 UnifiedEsWriter（管理 traces/events/logs 三个索引）
    try:
        writer = UnifiedEsWriter.get_instance()
        await writer.start()
    except Exception as exc:
        logger.warning("UnifiedEsWriter 启动失败（非致命）: %s", exc,
                       _type="system", event_name="es_writer_start", message=f"ES Writer 启动失败: {exc}", status="error")

    # 启动数据库 + 迁移 + Worker
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: Base.metadata.create_all(bind=engine))
    await loop.run_in_executor(None, lambda: migration_manager.run_all(engine))

    # Worker 内部管理生命周期
    await worker.start()
    logger.info("Agent Worker 已启动", _type="system", event_name="worker_start", message="Agent Worker 已启动", status="started")
    yield

    # 关闭 UnifiedEsWriter
    try:
        writer = UnifiedEsWriter.get_instance()
        await writer.stop()
    except Exception as exc:
        logger.warning("UnifiedEsWriter 停止错误: %s", exc,
                       _type="system", event_name="es_writer_stop", message=f"ES Writer 停止错误: {exc}", status="error")

    await worker.stop()


app = FastAPI(
    title="AutoPublish API",
    description="AI 内容生产与发布工具 API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# 允许的前端源
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:9010",
    "http://127.0.0.1:9010",
]


# ─────────────────────────────────────────────────────────────────────────────
# 异常处理器 — 按优先级注册（FastAPI 匹配先注册的，再匹配更具体的子类）。
# 处理器必须在中间件和路由之前注册，装饰器才能正确生效。
# ─────────────────────────────────────────────────────────────────────────────


# 1.  ValidationError — 422
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    logger.warning("参数校验失败", message=exc.message,
                   _type="http", event_name="validation_error",
                   path=request.url.path, details=exc.details, status=422)
    resp = build_error_response(exc, request, 422, exc.error_code)
    return JSONResponse(**resp)


# 2.  RequestValidationError — 422（FastAPI 内置，Pydantic）
@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    field_errors = {}
    for err in errors:
        loc = ".".join(str(l) for l in err["loc"])
        field_errors[loc] = err["msg"]

    details = {"fields": field_errors}
    logger.warning("参数校验失败", field_errors=field_errors,
                   _type="http", event_name="request_validation_error",
                   path=request.url.path, status=422)
    resp = build_error_response(exc, request, 422, "VALIDATION_ERROR", details)
    return JSONResponse(**resp)


# 3.  AuthenticationError — 401
@app.exception_handler(AuthenticationError)
async def authentication_error_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
    logger.warning("认证失败", message=exc.message,
                   _type="http", event_name="authentication_error",
                   path=request.url.path, status=401)
    resp = build_error_response(exc, request, 401, exc.error_code)
    resp["headers"]["WWW-Authenticate"] = "Bearer"
    return JSONResponse(**resp)


# 4.  AuthorizationError — 403
@app.exception_handler(AuthorizationError)
async def authorization_error_handler(request: Request, exc: AuthorizationError) -> JSONResponse:
    logger.warning("权限不足", message=exc.message,
                   _type="http", event_name="authorization_error",
                   path=request.url.path, details=exc.details, status=403)
    details = dict(exc.details)
    if "required_permission" not in details:
        details["required_permission"] = None
    resp = build_error_response(exc, request, 403, exc.error_code, details)
    return JSONResponse(**resp)


# 5.  NotFoundError — 404
@app.exception_handler(NotFoundError)
async def not_found_error_handler(request: Request, exc: NotFoundError) -> JSONResponse:
    logger.warning("资源不存在", message=exc.message,
                   _type="http", event_name="not_found_error",
                   path=request.url.path, status=404)
    resp = build_error_response(exc, request, 404, exc.error_code)
    return JSONResponse(**resp)


# 6.  ConflictError — 409
@app.exception_handler(ConflictError)
async def conflict_error_handler(request: Request, exc: ConflictError) -> JSONResponse:
    logger.warning("资源冲突", message=exc.message,
                   _type="http", event_name="conflict_error",
                   path=request.url.path, details=exc.details, status=409)
    resp = build_error_response(exc, request, 409, exc.error_code)
    return JSONResponse(**resp)


# 7.  RateLimitError — 429
@app.exception_handler(RateLimitError)
async def rate_limit_error_handler(request: Request, exc: RateLimitError) -> JSONResponse:
    logger.warning("请求过于频繁", message=exc.message,
                   _type="http", event_name="rate_limit_error",
                   path=request.url.path, status=429)
    resp = build_error_response(exc, request, 429, exc.error_code)
    return JSONResponse(**resp)


# 8.  HTTPException — FastAPI 原生（任意状态码）
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning("HTTP 异常", status_code=exc.status_code, detail=exc.detail,
                   _type="http", event_name="http_exception",
                   path=request.url.path, status=exc.status_code)
    resp = build_error_response(
        exc, request, exc.status_code, f"HTTP_{exc.status_code}",
        details={"detail": exc.detail}
    )
    return JSONResponse(**resp)


# 9.  ServiceError — 5xx 服务层错误（GenerateError, AgentError 等）
@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    logger.error("服务层异常", message=exc.message, exc_type=type(exc).__name__,
                 _type="service", event_name="service_error",
                 path=request.url.path, status=exc.status_code,
                 error_type=type(exc).__name__, exc_info=True)
    resp = build_error_response(exc, request, exc.status_code, exc.error_code)
    return JSONResponse(**resp)


# 10.  WebSocketDisconnect — 静默透传（非错误）
@app.exception_handler(WebSocketDisconnect)
async def websocket_disconnect_handler(request: Request, exc: WebSocketDisconnect) -> JSONResponse:
    pass  # FastAPI 通过关闭 WebSocket 连接来处理


# 11.  Exception — 最终兜底（500）
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    exc_type = type(exc).__name__
    logger.error("未处理异常", exc=exc, exc_type=exc_type,
                 _type="system", event_name="unhandled_error",
                 path=request.url.path, status=500,
                 error_type=exc_type, exc_info=True)

    # 构建经过清理的响应（不泄露内部细节）
    sanitised = exc.__class__.__name__
    resp = build_error_response(
        exc, request, 500, "INTERNAL_ERROR",
        details={"type": sanitised}
    )
    # 覆盖消息以隐藏内部堆栈跟踪
    resp["content"]["error"]["message"] = "发生内部错误，请稍后重试。"
    return JSONResponse(**resp)


# CORS 中间件（开发模式允许所有来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 请求日志中间件（统一结构化日志 + trace_id）
from src.logging_shared.middleware import LoggingMiddleware
app.add_middleware(LoggingMiddleware, service_name="autopublish-api")

# 注册路由
app.include_router(auth.router, prefix="/api/v1")
app.include_router(contents.router, prefix="/api/v1")
app.include_router(publish.router, prefix="/api/v1")
app.include_router(platforms.router, prefix="/api/v1")
app.include_router(presets.router, prefix="/api/v1")
app.include_router(ai_editor.router, prefix="/api/v1")
app.include_router(personas.router, prefix="/api/v1")
app.include_router(admin_router)
app.include_router(skills_router)
app.include_router(chat_router)
app.include_router(agent_router, prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "AutoPublish API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """健康检查，附带子系统验证。"""
    from src.database import SessionLocal

    db_healthy = False
    try:
        def _ping():
            from sqlalchemy import text
            db = SessionLocal()
            try:
                db.execute(text("SELECT 1"))
                return True
            finally:
                db.close()
        db_healthy = await asyncio.get_running_loop().run_in_executor(None, _ping)
    except Exception as e:
        logger.warning("健康检查数据库 Ping 失败: %s", e)

    worker_healthy = worker._running if hasattr(worker, '_running') else False

    all_healthy = db_healthy and worker_healthy
    status = "healthy" if all_healthy else "degraded"

    return {
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "database": "passed" if db_healthy else "failed",
            "worker": "running" if worker_healthy else "stopped",
        },
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标端点 — 真实的 prometheus_client 指标。"""
    from fastapi.responses import Response
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=9000)
