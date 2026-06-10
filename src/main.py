import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import asyncio

from src.core.exceptions import GenerateError
from src.core.logging import setup_logging
from src.core.migrations import migration_manager
from src.config import get_settings
from src.database import engine
from src.models import Base
from src.api import auth, contents, publish, platforms, templates, presets, ai_editor, personas
from src.api.skills import router as skills_router
from src.api.chat import router as chat_router
from src.agent.api import router as agent_router
from src.agent.workflow.worker import worker
from src.observability import setup_observability
from src.observability.api import router as observability_router
from src.observability.es_client import close_es
from src.logging_shared.unified_writer import UnifiedEsWriter
import src.migrations  # noqa: F401 — register migrations

import structlog
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Unified logging configuration (reads settings.log_level, settings.log_json_format)
    setup_logging(settings)

    # 统一 tracing 初始化（structlog only，OTel SDK 已移除）
    tracer, meter = setup_observability(
        service_name="autopublish-api",
        service_version="2.0.0",
        settings=settings,
    )

    # 启动 UnifiedEsWriter（管理 traces/events/logs 三个索引）
    try:
        writer = UnifiedEsWriter.get_instance()
        await writer.start()
    except Exception as exc:
        logger.warning("UnifiedEsWriter start failed (non-fatal): %s", exc)

    # Start DB + migrations + worker
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: Base.metadata.create_all(bind=engine))
    await loop.run_in_executor(None, lambda: migration_manager.run_all(engine))

    # Worker handles lifecycle internally
    await worker.start()
    logger.info("Agent Worker started")
    yield

    # 关闭 UnifiedEsWriter
    try:
        writer = UnifiedEsWriter.get_instance()
        await writer.stop()
    except Exception as exc:
        logger.warning("UnifiedEsWriter stop error: %s", exc)

    await close_es()
    await worker.stop()
    logger.info("Agent Worker stopped")


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


def _get_cors_origin(request: Request) -> str:
    origin = request.headers.get("origin", "")
    return origin if origin in ALLOWED_ORIGINS else ALLOWED_ORIGINS[0]


@app.exception_handler(GenerateError)
async def generate_error_handler(request: Request, exc: GenerateError) -> JSONResponse:
    logger.error("GenerateError: %s", exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": {"message": exc.message, "type": "GenerateError"}},
        headers={
            "Access-Control-Allow-Origin": _get_cors_origin(request),
            "Access-Control-Allow-Credentials": "true",
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {"message": str(exc), "type": type(exc).__name__}
        },
        headers={
            "Access-Control-Allow-Origin": _get_cors_origin(request),
            "Access-Control-Allow-Credentials": "true",
        },
    )


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
app.include_router(templates.router, prefix="/api/v1")
app.include_router(ai_editor.router, prefix="/api/v1")
app.include_router(personas.router, prefix="/api/v1")
app.include_router(skills_router)
app.include_router(chat_router)
app.include_router(agent_router, prefix="/api/v1")
app.include_router(observability_router)


@app.get("/")
def root():
    return {"message": "AutoPublish API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check with subsystem verification."""
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
        logger.warning("Health check DB ping failed: %s", e)

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
    """Prometheus metrics endpoint - returns basic metrics in Prometheus text format."""
    from fastapi.responses import Response

    # Basic metrics in Prometheus text format
    metrics_text = """# HELP autopublish_up Whether the service is up (1) or down (0)
# TYPE autopublish_up gauge
autopublish_up 1

# HELP autopublish_info Service info
# TYPE autopublish_info gauge
autopublish_info{version="2.0.0"} 1
"""
    return Response(content=metrics_text, media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=9000)
