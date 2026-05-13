from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime

from src.core.exceptions import GenerateError
from src.database import engine
from src.models import Base
from src.api import auth, contents, publish, platforms, templates, presets, ai_editor, personas
from src.agent.api import router as agent_router
from src.agent.workflow.worker import worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：创建表 + 启动 Worker
    Base.metadata.create_all(bind=engine)
    await worker.start()
    print("[Agent Worker] Started")
    yield
    # 关闭时：停止 Worker
    await worker.stop()
    print("[Agent Worker] Stopped")


app = FastAPI(
    title="AutoPublish API",
    description="AI 内容生产与发布工具 API",
    version="2.0.0",  # Agent 架构升级
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
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def _get_cors_origin(request: Request) -> str:
    origin = request.headers.get("origin", "")
    return origin if origin in ALLOWED_ORIGINS else ALLOWED_ORIGINS[0]


@app.exception_handler(GenerateError)
async def generate_error_handler(request: Request, exc: GenerateError) -> JSONResponse:
    print(f"[ERROR] GenerateError: {exc.message}")
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
    import traceback
    print(f"[ERROR] Unhandled exception: {exc}")
    traceback.print_exc()
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
app.include_router(agent_router, prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "AutoPublish API", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000)
