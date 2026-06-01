import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import asyncio
import os

from src.core.exceptions import GenerateError
from src.core.migrations import migration_manager
from src.config import get_settings
from src.database import engine
from src.models import Base
from src.api import auth, contents, publish, platforms, templates, presets, ai_editor, personas
from src.api.skills import router as skills_router
from src.agent.api import router as agent_router
from src.agent.workflow.worker import worker
from src.observability import setup_observability, setup_local_dev
from src.observability import metrics as obs_metrics
from src.observability.api import router as observability_router
from src.observability.es_client import close_es
import src.migrations  # noqa: F401 — register migrations

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log_json = os.getenv("LOG_JSON", "true").lower() == "true"
    use_local_dev = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip() == ""

    if use_local_dev:
        tracer, meter = setup_local_dev()
    else:
        tracer, meter = setup_observability(
            service_name="autopublish-api",
            service_version="2.0.0",
            log_json=log_json,
            debug=settings.debug,
        )

    obs_metrics.create_meters()

    # Start DB + migrations + worker
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: Base.metadata.create_all(bind=engine))
    await loop.run_in_executor(None, lambda: migration_manager.run_all(engine))

    # Start Gateway 管线（worker 内也会调用，幂等安全）
    from src.core.gateway.core.container import GatewayContainer
    await GatewayContainer.get_instance()
    logger.info("GatewayContainer started (lifespan)")

    await worker.start()
    logger.info("Agent Worker started")
    yield

    await close_es()
    await worker.stop()
    logger.info("Agent Worker stopped")
    # GatewayContainer.stop() 已在 worker.stop() 中调用


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
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
    "http://localhost:5176",
    "http://127.0.0.1:5176",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000)
