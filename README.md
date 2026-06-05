# AutoPublish Backend

AI 内容生产与发布工具后端 API 服务，基于 FastAPI + LangGraph 多 Agent 编排引擎。所有 LLM 调用统一经过 LLM Gateway 服务。

## 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| **Web 框架** | FastAPI | 异步 API + 自动文档 |
| **Agent 编排** | LangGraph | StateGraph DAG 调度 |
| **LLM 调用** | LangChain + LLM Gateway | httpx → Gateway → 多供应商 |
| **多供应商** | OpenAI / 智谱 / 通义千问 | Gateway 统一封装 |
| **数据库** | SQLAlchemy + SQLite | 内容、用户、模板存储 |
| **认证** | Python-JOSE + Passlib | JWT Token |
| **外部工具** | MCP (Model Context Protocol) | 搜索、爬虫等工具标准化 |
| **可观测性** | OpenTelemetry + Prometheus + Elasticsearch | 全链路追踪 + 指标采集 + 结构化日志 |
| **异步队列** | asyncio.Queue | 内存任务队列 |
| **实时推送** | SSE + WebSocket | 任务进度实时推送 |
| **日志** | structlog (统一配置) | 中文结构化日志，Console + ES |

## 快速开始

```bash
# 1. 启动 LLM Gateway（必须先启动）
cd llm-gateway
uvicorn gateway.main:app --port 8001

# 2. 启动后端
cd backend
pip install -r requirements.txt
cp .env.example .env    # 编辑 .env 填入实际配置
uvicorn src.main:app --reload --port 8000
```

API 文档：http://localhost:8000/docs
LLM Gateway：http://localhost:8001/docs

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                      前端层 Frontend                                  │
│  React 18 + TypeScript + Vite + Tailwind CSS                         │
│  Zustand 状态管理 + React Router                                      │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ HTTP REST / SSE
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    API 层 (FastAPI)                                    │
│  /api/v1/auth/*  /api/v1/contents/*  /api/v1/skills/*              │
│  /api/v1/publish/*  /api/v1/ai-editor/*                             │
│  /api/v1/agent/*  /agent/stream  /agent/ws/{id}                      │
│                                                                      │
│  LoggingMiddleware ──► 统一结构化日志 + trace_id                       │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│                  Agent 编排层 (LangGraph)                               │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Orchestrator                               │  │
│  │  任务拆解 + 调度中枢 + 结果合并                               │  │
│  └────────────────────────────┬─────────────────────────────────┘  │
│                               │                                       │
│  ┌────────────────────────────▼─────────────────────────────────┐  │
│  │                    LangGraph StateGraph                       │  │
│  │  classify → schedule → execute → reflect → merge            │  │
│  └────────────────────────────┬─────────────────────────────────┘  │
│                               │                                       │
│  ┌────────────────────────────▼─────────────────────────────────┐  │
│  │                    核心 Agent 团队                             │  │
│  │  Manager │ Writing │ SEO │ Review │ Diagram │ WebSearch │ ...│  │
│  │  (全部通过 ToolRegistry 绑定 LLM，所有 LLM 调用经由 Gateway)   │  │
│  └────────────────────────────┬─────────────────────────────────┘  │
│                               │                                       │
│  ┌────────────────────────────▼─────────────────────────────────┐  │
│  │                    支撑系统 (可选)                              │  │
│  │  MessageBus │ ReflectionEngine │ ToolRegistry │ HITLManager   │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             │  LangChain ChatOpenAI (base_url=localhost:8001)
                             │  httpx (LLMGateway / generate_sync)
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│              LLM Gateway (8001) — 统一 LLM 出口                        │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Plugin Chain: RateLimiter → Cache → CircuitBreaker          │   │
│  │               → Observability                                 │   │
│  └────────────────────────────┬─────────────────────────────────┘   │
│                               │                                       │
│  ┌────────────────────────────▼─────────────────────────────────┐   │
│  │  Auto Router (三维加权打分：成本 / 延迟 / 质量)              │   │
│  │  或直接指定模型                                               │   │
│  └────────────────────────────┬─────────────────────────────────┘   │
│                               │                                       │
│  ┌────────────────────────────▼─────────────────────────────────┐   │
│  │  Provider Adapters (httpx 异步)                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐       │   │
│  │  │ OpenAI   │  │  智谱    │  │   通义千问            │       │   │
│  │  │ gpt-4o   │  │ glm-4    │  │   qwen-plus          │       │   │
│  │  └──────────┘  └──────────┘  └──────────────────────┘       │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  LoggingMiddleware ──► trace_id 透传 ──► 结构化日志                  │
│  ObservabilityPlugin ──► LLM 请求详情 ──► Console + ES 异步写入     │
└──────────────────────────────────────────────────────────────────────┘
```

### 关键变化：LLM 调用链路

| 变更前 | 变更后 |
|--------|--------|
| Backend 直接调用 LLM 提供商 | Backend → LLM Gateway (8001) → LLM 提供商 |
| 后端各自记录 LLM 日志 | Gateway 统一记录，后端透传 trace_id |
| 无统一限流/缓存 | Gateway 共享 RateLimiter + Cache + CircuitBreaker |
| 各服务独立日志配置 | 共享 `logging_shared` 统一配置 |

## LangGraph 编排流程

```
用户请求 → Orchestrator.run()
              │
              ▼
    classify_and_decompose ────► 任务复杂度分类 + LLM 动态拆解
              │                    (simple / medium / complex)
              │                    或模板兜底
              ▼
    schedule_and_execute ────► DAG 依赖分析 + 并行批次调度
              │
              ▼
    execute_task_batch ────► asyncio.gather 并行执行
              │
              ├─ [可选] reflect_on_task_batch ───► 三层反思 (Self/Peer/Meta)
              │                                        │
              ▼                                        ▼
         merge_and_finish ◄────────────────────────────┘
              │
              ▼
         返回结果
```

### 可选 Phase 特性

| Phase | 特性 | 开关 | 说明 |
|-------|------|------|------|
| **Phase 1** | Agent Registry | `enable_agent_registry` | Agent 注册与工厂模式 |
| **Phase 2** | Message Bus | `enable_message_bus` | Agent 间 P2P 通信 |
| **Phase 3** | Reflection | `enable_reflection` | 自省/同行评审/元认知 |
| **Phase 4** | HITL | `enable_hitl` | 人工介入决策门控 |

## 异步工作流与 SSE

```
Client → POST /agent/stream
           │
           ├─ ① 生成执行计划 (预计算)
           ├─ ② enqueue → TaskQueue
           ├─ ③ AgentWorker 出队执行
           │     ├─ orchestrator.run() (LangGraph)
           │     ├─ LLM 调用经由 Gateway
           │     ├─ EventBus 推送进度
           │     └─ 结果 → complete_task
           └─ ④ SSE 实时推送事件
                 event: plan / task_created
                 event: thinking_step / reasoning / llm_chunk
                 event: progress / decision_required
                 event: result / error / done
```

## Agent 系统

### 已注册 Agent（12 个）

| Agent | 功能 | 实现 |
|-------|------|------|
| **ManagerAgent** | 任务拆解 + 调度中枢 | LangGraph 编排器 |
| **WritingAgent** | 多风格内容创作 | LLM + 模板 |
| **SEOAgent** | SEO 评分、关键词优化 | LLM + 评分算法 |
| **ReviewAgent** | 质量审核、违禁词检查 | LLM + 违禁词库 |
| **DiagramAgent** | Mermaid 图表生成 | LLM |
| **ImageAgent** | 配图生成 | LLM |
| **HotTopicAgent** | 热点抓取分析 | httpx |
| **CodeAgent** | 代码片段生成 | LLM |
| **WebSearch** | 网络搜索 | MCP: WebSearch Server |
| **PlatformAdapter** | 多平台格式转换 | 平台适配器 |
| **TranslationAgent** | 多语言翻译 | LLM |
| **FactCheckAgent** | 事实核查 | LLM + 搜索验证 |

### MCP 工具层

| Server | 功能 | 默认启用 | 需要配置 |
|--------|------|---------|----------|
| **WebSearch Server** | 网页/新闻搜索 | ✅ | - |
| **Jina Reader Server** | 网页 Markdown 抓取 | ✅ | - |
| **Brave Search Server** | 隐私优先搜索 | ❌ | `BRAVE_API_KEY` |
| **Tavily Search Server** | AI 优化搜索 | ❌ | `TAVILY_API_KEY` |
| **Pexels Server** | 免费高清图片 | ❌ | `PEXELS_API_KEY` |
| **GitHub Server** | 代码/Issue 搜索 | ❌ | `GITHUB_TOKEN` |

## 项目结构

```
backend/
├── src/
│   ├── main.py                 # FastAPI 入口 + lifespan（接入 LoggingMiddleware + ES writer）
│   ├── config.py               # Pydantic Settings 配置管理
│   ├── database.py             # SQLAlchemy 引擎 + SessionLocal
│   ├── models.py               # ORM 模型
│   ├── logging_shared/         # 统一日志基础设施（两服务共用）
│   │   ├── fields.py           # 日志字段常量
│   │   ├── config.py           # structlog 统一配置
│   │   ├── middleware.py       # LoggingMiddleware（HTTP 请求日志 + trace_id）
│   │   └── es_writer.py        # ES 异步批量写入器
│   ├── api/                    # API 路由层
│   │   ├── auth.py             # JWT 认证
│   │   ├── contents.py         # 内容 CRUD + 日历
│   │   ├── publish.py          # 发布管理
│   │   ├── platforms.py        # 平台管理
│   │   ├── templates.py         # Prompt 模板
│   │   ├── presets.py           # 用户预设
│   │   ├── personas.py          # 人设管理
│   │   ├── ai_editor.py         # AI 编辑助手
│   │   └── skills.py            # Skill CRUD + 导入
│   ├── agent/                  # Agent 系统
│   │   ├── orchestrator.py     # 编排器 (LangGraph 包装)
│   │   ├── graph.py             # LangGraph StateGraph
│   │   ├── nodes.py             # 图节点函数
│   │   ├── edges.py             # 条件边路由
│   │   ├── state.py             # AgentGraphState
│   │   ├── factory.py           # 工厂函数（创建 Orchestrator）
│   │   ├── api.py               # Agent REST/SSE/WS 端点
│   │   ├── schemas.py           # AgentTask / AgentResult
│   │   ├── thinking_trace.py    # 思考链路
│   │   ├── persistence.py       # 持久化
│   │   ├── cache.py             # 结果缓存
│   │   ├── retry.py             # 重试策略
│   │   ├── agents/              # Agent 基类与注册表
│   │   ├── handlers/            # Agent 处理函数
│   │   ├── reflection/          # 三层反思引擎
│   │   ├── communication/        # Agent 间通信
│   │   ├── hitl/                # Human-in-the-Loop
│   │   ├── tools/               # Agent 工具系统
│   │   └── workflow/            # 异步任务队列 + Worker
│   ├── mcp/                     # MCP (Model Context Protocol)
│   ├── memory/                  # 记忆系统
│   ├── skills/                  # Skill 内置注册表
│   ├── core/                    # 核心模块
│   │   ├── llm_gateway.py      # LLM Gateway 客户端（httpx 调用 Gateway）
│   │   ├── model_router.py      # 按角色创建 ChatOpenAI（指向 Gateway）
│   │   ├── prompt_engine.py     # Prompt 引擎
│   │   ├── content_service.py   # 内容生成服务
│   │   └── security.py          # 安全认证
│   ├── adapters/                # 平台适配器
│   ├── utils/
│   ├── prompts/                 # Prompt 模板
│   └── observability/            # 可观测性
│       ├── tracing.py
│       ├── metrics.py
│       └── es_client.py
├── tests/
├── scripts/
├── docs/
├── .env.example
└── requirements.txt
```

## LLM 调用

### 架构说明

所有 LLM 调用统一经过 LLM Gateway（8001），不再直连 LLM 提供商：

```
Backend                    LLM Gateway (8001)            LLM 提供商
  │                              │                           │
  │  ModelRouter                 │                           │
  │  ChatOpenAI(                 │                           │
  │    base_url=8001)            │                           │
  │  ─────────────────────────► │  RateLimiter               │
  │                              │  Cache                     │
  │                              │  CircuitBreaker            │
  │                              │  Observability             │
  │                              │  ─────────────────────────►│
  │                              │  ◄─────────────────────────│
  │                              │                           │
  │  ◄───────────────────────── │                           │
  │                              │                           │
```

### 两种调用方式

**1. Agent 工具调用（LangChain）：** ModelRouter 返回 ChatOpenAI 实例，base_url 指向 Gateway，LangChain 自动处理流式响应和工具调用。

```python
from src.core.model_router import ModelRouter

router = ModelRouter()
llm = router.get_chat_model("WritingAgent")
response = await llm.ainvoke(messages)  # 经由 Gateway
```

**2. 简单 LLM 调用（httpx）：** `LLMGateway.generate_sync()` 用于 Orchestrator 的计划拆解等简单调用。

```python
from src.core.llm_gateway import LLMGateway

gateway = LLMGateway(api_key="...")
content = await gateway.generate_sync(messages)  # 经由 Gateway
```

### 多供应商支持

LLM Gateway 自动按优先级选择可用的供应商：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ZHIPU_API_KEY` | 智谱 GLM | `glm-4-flash` |
| `QWEN_API_KEY` | 通义千问 | `qwen-plus` |
| `OPENAI_API_KEY` | OpenAI | `gpt-4o-mini` |

默认优先级：`ZHIPU_API_KEY` → `QWEN_API_KEY` → `OPENAI_API_KEY`

### 按角色差异化配置

支持为不同 Agent 角色指定不同模型：

| 角色 | 说明 | 默认模型 |
|------|------|----------|
| `ManagerAgent` | 任务拆解与规划 | 智谱 glm-4-flash |
| `WritingAgent` | 内容生成 | 智谱 glm-4-flash |
| `SEOAgent` | SEO 优化 | 智谱 glm-4-flash |
| `ReviewAgent` | 质量审核 | 智谱 glm-4-flash |
| `DiagramAgent` | 图表生成 | 通义 qwen-plus |
| `WebSearch` | 搜索分析 | 智谱 glm-4-flash |

## 统一日志

后端和 LLM Gateway 共用 `src/logging_shared/` 中的统一配置：

- **格式：** 中文结构化 JSON（生产）或彩色 Console（开发）
- **trace_id：** HTTP header `X-Trace-Id` 跨服务透传，串联所有日志
- **ES 写入：** 异步批量写入 `logs-YYYY.MM.DD` 索引，不阻塞请求
- **事件类型：** `http_request_start/end` | `llm_request_start/end/error`

```
# 示例：llm_request_end 日志字段
{
  "trace_id": "a1b2c3d4e5f60718",
  "event": "llm_request_end",
  "service": "llm-gateway",
  "llm_model": "glm-4-flash",
  "llm_provider": "zhipu",
  "llm_latency_ms": 1234.5,
  "llm_prompt_tokens": 256,
  "llm_completion_tokens": 128,
  "llm_total_tokens": 384,
  "status_code": 200
}
```

## API 概览

### Agent 接口

| 端点 | 说明 |
|------|------|
| `POST /api/v1/agent/run` | 同步执行 Agent 编排 |
| `POST /api/v1/agent/plan` | 预览执行计划 |
| `POST /api/v1/agent/stream` | SSE 流式执行 |
| `POST /api/v1/agent/async/submit` | 提交异步任务 |
| `GET /api/v1/agent/async/status/{id}` | 查询任务状态 |
| `GET /api/v1/agent/async/list` | 任务列表 |
| `DELETE /api/v1/agent/async/{id}` | 取消任务 |
| `POST /api/v1/agent/decision/{id}` | 提交用户决策 |
| `POST /api/v1/agent/repurpose` | 内容再利用 Pipeline |
| `WS /api/v1/agent/ws/{id}` | WebSocket 实时进度 |
| `GET /api/v1/agent/dlq` | 死信队列 |
| `POST /api/v1/agent/dlq/{id}/retry` | 重试死信任务 |
| `GET /api/v1/agent/metrics` | Agent 监控指标 |

### 内容接口

| 端点 | 说明 |
|------|------|
| `GET/POST /api/v1/contents` | 内容列表 / 新建 |
| `GET/PUT/DELETE /api/v1/contents/{id}` | 内容 CRUD |
| `POST /api/v1/contents/{id}/generate` | AI 生成内容 |
| `POST /api/v1/contents/{id}/generate-stream` | 流式生成 |
| `POST /api/v1/contents/batch-generate` | 批量生成 |
| `GET /api/v1/contents/{id}/versions` | 版本历史 |
| `GET /api/v1/contents/{id}/versions/diff` | 版本对比 |
| `POST /api/v1/contents/{id}/check-banned-words` | 违禁词检查 |
| `GET /api/v1/contents/calendar` | 内容日历 |

### Skills 接口

| 端点 | 说明 |
|------|------|
| `GET /api/v1/skills` | 列出用户所有 Skill |
| `GET /api/v1/skills/{id}` | 获取 Skill 详情 |
| `POST /api/v1/skills` | 创建新 Skill |
| `PUT /api/v1/skills/{id}` | 更新 Skill |
| `DELETE /api/v1/skills/{id}` | 删除 Skill |
| `GET /api/v1/skills/builtins` | 列出内置 Skill |
| `POST /api/v1/skills/import/url` | 从 URL 导入 |
| `POST /api/v1/skills/import/file` | 从文件导入 |

### 发布与 AI 编辑

| 端点 | 说明 |
|------|------|
| `POST /api/v1/contents/{id}/publish` | 发布到平台 |
| `POST /api/v1/publish/batch` | 批量发布 |
| `GET /api/v1/platforms` | 平台列表 |
| `POST /api/v1/ai/generate` | AI 内容生成 |
| `POST /api/v1/ai/edit` | AI 编辑优化 |
| `POST /api/v1/ai/translate` | AI 翻译 |
| `POST /api/v1/ai/check-banned-words` | AI 违禁词检查 |
| `POST /api/v1/ai/analyze-seo` | SEO 分析 |

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DATABASE_URL` | 数据库连接 | `sqlite:///./autopublish.db` |
| `SECRET_KEY` | JWT 密钥 | (本地开发密钥) |
| **LLM 配置** | | |
| `ZHIPU_API_KEY` | 智谱 API Key | - |
| `ZHIPU_MODEL` | 智谱模型 | `glm-4-flash` |
| `QWEN_API_KEY` | 通义千问 Key | - |
| `QWEN_MODEL` | 通义千问模型 | `qwen-plus` |
| `OPENAI_API_KEY` | OpenAI Key | - |
| `OPENAI_MODEL` | OpenAI 模型 | `gpt-4o-mini` |
| **Gateway 地址** | | |
| `GATEWAY_URL` | LLM Gateway URL | `http://localhost:8001` |
| **MCP 配置** | | |
| `MCP_ENABLED` | 启用 MCP | `true` |
| `MCP_ENABLE_SEARCH` | WebSearch Server | `true` |
| `MCP_ENABLE_JINA` | Jina Reader | `true` |
| `MCP_ENABLE_BRAVE` | Brave Search | `false` |
| `MCP_ENABLE_TAVILY` | Tavily Search | `false` |
| `MCP_ENABLE_PEXELS` | Pexels | `false` |
| `MCP_ENABLE_GITHUB` | GitHub | `false` |
| **可选功能** | | |
| `ENABLE_REFLECTION` | 启用反思引擎 | `false` |
| `ENABLE_MESSAGE_BUS` | 启用 Agent 通信 | `false` |
| `ENABLE_HITL` | 启用人工介入 | `false` |
| **日志配置** | | |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `LOG_JSON_FORMAT` | JSON 格式输出 | `false` |
| `ELASTICSEARCH_URL` | ES 地址 | `http://localhost:9200` |
| **平台凭证** | | |
| `XHS_COOKIE` | 小红书凭证 | - |
| `WECHAT_APP_ID` | 公众号 App ID | - |
| `WECHAT_APP_SECRET` | 公众号 Secret | - |
| `ZHIHU_COOKIE` | 知乎凭证 | - |

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行 Agent 测试
pytest tests/agents/ -v

# 查看 Agent 指标
curl http://localhost:8000/api/v1/agent/metrics

# 健康检查
curl http://localhost:8000/health
curl http://localhost:8001/health
```

## 关键设计决策

| 决策 | 方案 | 权衡 |
|------|------|------|
| **LLM 调用** | 统一经过 LLM Gateway | 集中治理（日志/限流/熔断/缓存），后端更轻量 |
| **编排引擎** | LangGraph StateGraph | 结构化 DAG，可预测性强 |
| **任务队列** | asyncio.Queue (内存) | 零依赖，单进程零开销 |
| **工具集成** | MCP (Model Context Protocol) | 标准化热插拔协议 |
| **模型路由** | ModelRouter 按角色创建 | 支持差异化配置和缓存 |
| **日志架构** | shared-logging 统一配置 + ES 异步写入 | trace_id 跨服务串联，日志不阻塞请求 |
| **可选 Phase** | 特性开关 (enable_*) | 按需启用，避免复杂度 |

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v2.1.0 | 2026-06 | 统一经 LLM Gateway 调用；shared-logging 统一日志；ES 异步写入 |
| v2.0.0 | 2026-06 | 重构 LLM 调用层，使用 langchain_openai 直接调用 |
| v1.x | 早期 | 初始架构，litellm + Model Gateway |
