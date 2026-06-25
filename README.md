# AutoPublish Backend

AI 内容生产与发布工具后端 API 服务，基于 FastAPI + LangGraph 多 Agent 编排引擎。所有 LLM 调用统一经过 LLM Gateway 服务（端口 9010）。

## 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| **Web 框架** | FastAPI | 异步 API + 自动文档 |
| **Agent 编排** | LangGraph | StateGraph DAG，5 节点 |
| **LLM 调用** | LangChain (ChatOpenAI) | 统一指向 LLM Gateway |
| **LLM 网关** | LLM Gateway (:9010) | AutoRouter 智能路由 + 限流/熔断/缓存 |
| **多供应商** | OpenAI / 智谱 / 通义千问 / DeepSeek | 通过 Gateway 统一封装 |
| **数据库** | SQLAlchemy + SQLite/PostgreSQL | 内容、用户、任务存储 |
| **认证** | Python-JOSE + Passlib | JWT Token |
| **外部工具** | MCP (Model Context Protocol) | langchain-mcp-adapters |
| **可观测性** | structlog + UnifiedEsWriter | 三层索引：traces / agent-events / logs |
| **异步队列** | asyncio.Queue | 内存任务队列 |
| **实时推送** | SSE + EventBus | 事件驱动的实时推送 |

## 快速开始

```bash
# 1. 启动 LLM Gateway（必须先启动）
cd llm-gateway
pip install -r requirements.txt
cp .env.example .env    # 编辑填入 API Key
uvicorn gateway.main:app --port 9010

# 2. 启动后端
cd ../backend
pip install -r requirements.txt
cp .env.example .env    # 编辑填入配置
uvicorn src.main:app --reload --port 9000
```

API 文档：http://localhost:9000/docs
LLM Gateway：http://localhost:9010/docs

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         前端层 Frontend                                  │
│  React 18 + TypeScript + Vite + Tailwind CSS                         │
│  Zustand + React Router                                                │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │ HTTP / SSE
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    API 层 (FastAPI, :9000)                               │
│  /api/v1/auth/*  /api/v1/contents/*  /api/v1/skills/*               │
│  /api/v1/chat/*  /api/v1/publish/*  /api/v1/agent/*                 │
│  LoggingMiddleware ──► trace_id + structlog                           │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │ model=auto / deepseek-chat
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  Agent 编排层 (LangGraph)                                   │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │              Orchestrator (run)                                    │   │
│  │  understand_intent → plan_tasks → schedule_and_execute           │   │
│  │    → execute_task_batch (asyncio.gather) → merge_and_finish      │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  Agent Handlers: Writing | Review | Fix | SEO | Diagram             │
│                   | Code | WebSearch | URLCrawler | HotTopic         │
│                                                                       │
│  Memory System: WorkingMemory | SemanticMemory | EpisodicMemory        │
│  MCP Client: Search | Brave | Jina | Tavily | Pexels | GitHub        │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    LLM Gateway (:9010)                                   │
│  POST /v1/chat/completions  │  GET /v1/models                         │
│  Plugin Chain: RateLimiter → Cache → Observability                     │
│  Router: AutoRouter (model=auto) + ExactMatch                        │
│  Adapters: OpenAI | Zhipu | Qwen | DeepSeek                          │
│  Circuit Breaker + Provider Fallback                                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

## LangGraph 编排流程

```
用户请求 → Orchestrator.run()
              │
              ▼
    understand_intent ───► LLM 意图分析（提取 intent/slots/platform）
              │                    规则兜底（LLM 不可用时）
              ▼
    plan_tasks ───► 任务拆解
              │        LLM 生成 / 规则兜底
              │        注入理解槽位 + 前端参数
              │        依赖链自动修正
              ▼
    schedule_and_execute ───► DAG 依赖分析
              │        拓扑排序，死锁检测
              │ (conditional: should_continue_or_finish)
              ▼
    execute_task_batch ───► asyncio.gather 并行执行
              │        semaphore 限流（默认 3 并发）
              │        Review-Fix 循环（最多 2 轮）
              │ (conditional: should_continue_or_finish)
              ▼
    merge_and_finish ───► FixAgent > ReviewFix > WritingAgent 优先级
              │        异步持久化到 DB
              ▼
         返回结果
```

## Agent 系统

### 已注册 Agent（10 个）

| Agent | Handler | 功能 | 模型策略 |
|-------|---------|------|---------|
| **WritingAgent** | `handle_writing` | 多风格内容创作 | auto-route (creative) |
| **ReviewAgent** | `handle_review` | 质量审核、违禁词检查 | auto-route (analysis) |
| **FixAgent** | `handle_fix` | 基于审核反馈优化内容 | auto-route (creative) |
| **SEOAgent** | `handle_seo` | SEO 评分、关键词优化 | auto-route (creative) |
| **DiagramAgent** | `handle_diagram` | Mermaid 图表生成 | auto-route (creative) |
| **CodeAgent** | `handle_code` | 代码片段生成 | 指定模型 (deepseek-chat) |
| **WebSearch** | `handle_web_search` | MCP 网络搜索 | auto-route (simple) |
| **URLCrawler** | `handle_url_crawl` | 页面内容抓取 | — |
| **HotTopicAgent** | `handle_hot_topic` | 热点抓取分析 | — |
| **ManagerAgent** | orchestrator 内 | 任务拆解与规划 | 指定模型 (deepseek-chat) |

### MCP 工具层

| Server | 功能 | 默认启用 | 需要配置 |
|--------|------|---------|---------|
| **WebSearch Server** | 网页/新闻搜索 | ✅ | — |
| **Jina Reader Server** | 网页 Markdown 抓取 | ✅ | — |
| **Brave Search Server** | 隐私优先搜索 | ❌ | `BRAVE_API_KEY` |
| **Tavily Search Server** | AI 优化搜索 | ❌ | `TAVILY_API_KEY` |
| **Pexels Server** | 免费高清图片 | ❌ | `PEXELS_API_KEY` |
| **GitHub Server** | 代码/Issue 搜索 | ❌ | `GITHUB_TOKEN` |

## 实时推送（SSE + EventBus）

```
Client → POST /agent/stream
           │
           ├─ enqueue → TaskQueue
           ├─ AgentWorker 出队执行
           │     ├─ orchestrator.run() (LangGraph)
           │     ├─ ModelRouter → ChatOpenAI → Gateway
           │     ├─ EventBus 推送事件
           │     └─ complete_task / fail_task
           └─ SSE 实时推送
                 event: plan / task_created
                 event: stage (started / completed)
                 event: thinking_step
                 event: llm_chunk / reasoning
                 event: progress
                 event: result / error / done
```

## 项目结构

```
backend/
├── src/
│   ├── main.py                 # FastAPI 入口 + lifespan（11 个异常处理器）
│   ├── config.py               # Pydantic Settings（含 per-role LLM 配置）
│   ├── database.py             # SQLAlchemy 引擎 + SessionLocal
│   ├── migrations.py            # DB 迁移注册
│   │
│   ├── api/                   # REST API 路由
│   │   ├── auth.py            # JWT 认证
│   │   ├── contents.py        # 内容 CRUD / 生成 / 日历 / 版本
│   │   ├── publish.py         # 多平台发布
│   │   ├── platforms.py        # 平台元数据
│   │   ├── presets.py          # 预设模板
│   │   ├── ai_editor.py       # AI 编辑器
│   │   ├── personas.py        # 角色预设
│   │   ├── skills.py          # Skill CRUD / 导入导出
│   │   ├── chat.py            # 对话历史（会话/消息 CRUD）
│   │   ├── conversation.py     # 对话引擎（intent 检测 / trigger）
│   │   ├── admin.py           # 管理路由
│   │   └── agent/
│   │       └── api.py          # Agent 运行 / 计划 / 重利用 / SSE 流
│   │
│   ├── agent/                 # Agent 系统
│   │   ├── orchestrator.py    # Orchestrator（LangGraph StateGraph 包装）
│   │   ├── graph.py           # build_agent_graph() — 5 节点 DAG
│   │   ├── nodes.py          # 全部节点函数（意图/规划/调度/执行/合并）
│   │   ├── state.py          # AgentGraphState TypedDict
│   │   ├── factory.py         # create_orchestrator()
│   │   ├── schemas.py         # AgentResult / AgentTask / ExecutionPlan
│   │   ├── result.py         # agent_response / error_response 构造器
│   │   ├── registry.py        # AgentCard 注册表 + 依赖规则
│   │   ├── stage_map.py      # Agent → 阶段标签/图标映射
│   │   ├── platform_registry.py  # 平台元数据（10 平台）
│   │   ├── platform_profile.py   # 平台写作 DNA（15 种内容形态）
│   │   ├── cache.py          # agent_cache / llm_cache（TTLCache 实例）
│   │   ├── persistence.py    # save_agent_task / save_subtask
│   │   ├── agent_utils.py    # extract_output / SYSTEM_PROMPT 等
│   │   ├── output_filter.py # strip_meta_dump / content_quality_check
│   │   └── workflow/
│   │       ├── worker.py      # AgentWorker（异步任务 + 重试策略）
│   │       └── task_queue.py  # TaskQueue / EventBus / AsyncTask
│   │
│   ├── handlers/              # Agent Handler 实现
│   │   ├── __init__.py       # HANDLER_MAP
│   │   ├── writing.py        # 多风格/多平台内容生成
│   │   ├── review.py         # 质量审核
│   │   ├── fix.py            # 基于审核反馈优化
│   │   ├── seo.py           # SEO 评分/关键词优化
│   │   ├── diagram.py       # Mermaid 图表生成
│   │   ├── code.py          # 代码片段生成
│   │   ├── web_search.py    # WebSearch / URLCrawler
│   │   ├── hot_topic.py     # 热点抓取分析
│   │   ├── tool_utils.py   # call_llm（带流式 chunk）
│   │   └── fallback.py      # handle_fallback
│   │
│   ├── memory/               # 记忆系统
│   │   ├── manager.py       # MemoryManager（统一入口）
│   │   ├── working_memory.py # WorkingMemory（会话内上下文）
│   │   ├── semantic_memory.py # SemanticMemory（向量语义存储）
│   │   ├── episodic_memory.py # EpisodicMemory（情景记忆）
│   │   ├── context_store.py  # ContextStore
│   │   ├── context_builder.py # ContextBuilder（多源上下文聚合）
│   │   ├── agent_scoped_view.py # 权限分片视图
│   │   └── tools.py        # run_memory_tool
│   │
│   ├── core/                 # 核心模块
│   │   ├── model_router.py  # ModelRouter（自动路由 + 指定模型）
│   │   ├── security.py      # JWT / Passlib / decode_token
│   │   ├── exceptions.py   # 统一异常类 + build_error_response
│   │   ├── request_guards.py # ensure_task_owner / validate_public_http_url
│   │   └── migrations.py   # DB migration_manager
│   │
│   ├── services/             # 业务服务层
│   │   ├── content_service.py
│   │   ├── content_parser_service.py
│   │   └── ai_editor_service.py
│   │
│   ├── prompts/              # Prompt 管理
│   │   ├── builder.py        # PromptBuilder（模块化构建）
│   │   ├── registry.py      # PromptRegistry
│   │   ├── engine.py        # PromptEngine
│   │   └── builtins/        # 内置 Prompt 模板
│   │
│   ├── mcp/                  # MCP (Model Context Protocol)
│   │   ├── client.py        # MCPServerConfig + MCPClient 多服务器管理
│   │   ├── web_search_server.py
│   │   ├── brave_search_server.py
│   │   ├── tavily_search_server.py
│   │   ├── jina_reader_server.py
│   │   ├── pexels_server.py
│   │   └── github_server.py
│   │
│   ├── conversation/          # 对话引擎
│   │   ├── engine.py        # ConversationEngine
│   │   ├── intent_detector.py # IntentDetector（trigger 生成）
│   │   └── schemas.py
│   │
│   ├── logging_shared/       # 统一日志基础设施
│   │   ├── config.py        # structlog 初始化
│   │   ├── unified_writer.py # UnifiedEsWriter（异步批量写 ES + JSONL 回退）
│   │   ├── middleware.py   # LoggingMiddleware（trace_id）
│   │   ├── fields.py        # 日志字段常量
│   │   ├── console.py      # Rich Console 输出
│   │   ├── agent_emitter.py # AgentEventEmitter
│   │   └── trace_context.py
│   │
│   ├── observability/         # 可观测性
│   │   ├── agent_tracer.py  # AgentTracer（轻量 span → UnifiedEsWriter）
│   │   ├── metrics.py      # Prometheus 指标（up_gauge / record_agent_run）
│   │   ├── tracing.py      # OTel（已简化为无 exporter）
│   │   └── es_client.py   # ES 客户端
│   │
│   └── tools/                # 工具函数
│       ├── vector_search.py  # FAISS 向量搜索
│       ├── code_runner.py   # 代码执行沙箱
│       ├── tokenizer.py     # Token 计数
│       └── langchain/
│           └── crawler_tool.py
│
├── llm-gateway/              # LLM Gateway 服务（:9010）
│   └── gateway/
│       ├── main.py           # FastAPI 入口，lifespan
│       ├── api/
│       │   ├── routes.py    # POST /v1/chat/completions
│       │   ├── schemas.py  # Pydantic 请求/响应模型
│       │   └── dependencies.py # build_adapters / build_router
│       ├── core/
│       │   ├── router.py   # AutoRouter（model=auto 路由策略）
│       │   ├── context.py  # RequestContext
│       │   └── exceptions.py # GatewayError / CircuitBreakerOpenError
│       ├── adapters/        # Provider 适配器
│       │   ├── base.py
│       │   └── openai.py
│       └── plugins/          # 中间件插件
│           ├── rate_limiter.py  # 滑动窗口限流
│           ├── cache.py        # Redis 响应缓存
│           ├── circuit_breaker.py # 状态机熔断
│           └── observability.py # 请求日志
│
├── tests/
├── .env.example
└── requirements.txt
```

## LLM 调用（统一路径）

### 架构说明

所有 LLM 调用统一经过 LLM Gateway（:9010），通过 LangChain ChatOpenAI：

```
Backend                      LLM Gateway (:9010)        LLM 提供商
  │                                │                      │
  │  ModelRouter                  │                      │
  │  get_chat_model(agent_type)   │                      │
  │  ChatOpenAI(base_url=:9010)  │                      │
  │  ──────────────────────────►  │  RateLimiter         │
  │                               │  Cache               │
  │                               │  CircuitBreaker      │
  │                               │  Observability       │
  │                               │  ─────────────────► │
  │                               │  ◄──────────────── │
  │  ◄─────────────────────────── │                      │
```

### ModelRouter 两种模式

```python
from src.core.model_router import ModelRouter

router = ModelRouter()

# 模式 1：自动路由 — model="auto" + X-Route-Hint
llm = router.get_chat_model("WritingAgent")
# → ChatOpenAI(model="auto", headers={"X-Route-Hint": "creative"})

# 模式 2：指定模型 — 用于高推理 Agent
llm = router.get_chat_model_for_model("deepseek-chat", "ManagerAgent")
# → ChatOpenAI(model="deepseek-chat")
```

### 按角色差异化配置

| 角色 | 模型策略 | 说明 |
|------|---------|------|
| `WritingAgent` | auto-route (creative) | 按成本选最便宜 provider |
| `ReviewAgent` | auto-route (analysis) | 分析任务 |
| `FixAgent` | auto-route (creative) | 创作优化 |
| `SEOAgent` | auto-route (creative) | SEO 优化 |
| `DiagramAgent` | auto-route (creative) | 图表生成 |
| `CodeAgent` | 指定模型 (deepseek-chat) | 代码生成 |
| `WebSearch` | auto-route (simple) | 搜索任务 |
| `ManagerAgent` | 指定模型 (deepseek-chat) | 任务规划 |

## 统一日志与可观测性

### 三层数据 → UnifiedEsWriter

| 数据类型 | 索引 | 数据来源 | 用途 |
|---------|------|---------|------|
| **Traces** | `traces-YYYY.MM.DD` | AgentTracer（轻量 span） | 链路追踪（JSON 结构写入 ES） |
| **Agent Events** | `agent-events-YYYY.MM.DD` | AgentEventEmitter | 推理过程、工具调用、成本 |
| **Logs** | `logs-YYYY.MM.DD` | structlog | 问题排查、审计 |

### ES 写入策略

- **异步批量写入**：队列缓冲 + 5s 刷新间隔或 batch=100 触发
- **JSONL 回退**：ES 不可用时写入 `logs/traces-events-fallback.jsonl`
- **非阻塞**：写入错误不中断主流程

### Prometheus 指标

```
/metrics  → up_gauge / info_gauge / record_agent_run / record_agent_task
```

## API 概览

### Agent 接口

| 端点 | 说明 |
|------|------|
| `POST /api/v1/agent/run` | 同步执行 Agent 编排（4 分钟超时） |
| `POST /api/v1/agent/plan` | 预览执行计划（不实际执行） |
| `POST /api/v1/agent/stream` | **SSE 流式执行**，事件驱动实时推送 |
| `POST /api/v1/agent/async/submit` | 提交异步任务到队列 |
| `GET /api/v1/agent/async/status/{id}` | 查询异步任务状态 |
| `GET /api/v1/agent/async/list` | 查询用户任务列表 |
| `DELETE /api/v1/agent/async/{id}` | 取消异步任务 |
| `POST /api/v1/agent/repurpose` | 内容再利用（一篇 → 多种形态） |
| `POST /api/v1/agent/crawl` | URL 抓取 + 可选 AI 分析 |
| `GET /api/v1/agent/dlq` | 死信队列列表 |
| `POST /api/v1/agent/dlq/{id}/retry` | 从 DLQ 重试任务 |
| `POST /api/v1/agent/vector-search` | 向量知识库搜索 |
| `GET /api/v1/agent/cache/stats` | 缓存统计 |
| `GET /api/v1/agent/metrics` | Agent 监控指标（实时 + DB 累计） |
| `WS /api/v1/agent/ws/{id}` | WebSocket 实时进度（JWT 鉴权） |

### 内容接口

| 端点 | 说明 |
|------|------|
| `POST /api/v1/contents/generate` | AI 内容生成（同步） |
| `POST /api/v1/contents/generate/stream` | AI 内容生成（SSE 流式） |
| `POST /api/v1/contents/generate/batch` | 批量生成多平台内容 |
| `GET /api/v1/contents` | 内容列表（含分页 + 全量统计） |
| `GET /api/v1/contents/calendar` | 内容日历（按天分组发布记录） |
| `GET/PUT/DELETE /api/v1/contents/{id}` | 内容 CRUD |
| `PUT /api/v1/contents/{id}/autosave` | 自动保存（不创建版本快照） |
| `POST /api/v1/contents/{id}/check-banned-words` | 敏感词检测 |
| `GET /api/v1/contents/{id}/versions` | 版本历史 |
| `GET /api/v1/contents/{id}/versions/diff` | 版本对比 |

### 对话接口

| 端点 | 说明 |
|------|------|
| `GET /api/v1/chat/conversations` | 对话列表（GROUP BY 优化，消除 N+1） |
| `POST /api/v1/chat/conversations` | 创建对话 |
| `GET /api/v1/chat/conversations/{id}` | 获取对话详情（含消息） |
| `PUT /api/v1/chat/conversations/{id}` | 更新对话标题 |
| `DELETE /api/v1/chat/conversations/{id}` | 删除对话（级联删除消息） |
| `POST /api/v1/chat/conversations/{id}/messages` | 添加消息 |
| `PUT /api/v1/messages/{id}` | 更新消息（更新状态/结果） |
| `DELETE /api/v1/messages/{id}` | 删除单条消息 |
| `POST /api/v1/chat/quick-start` | 快速开始（创建对话 + 首条消息） |

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

### AI 编辑器接口

| 端点 | 说明 |
|------|------|
| `POST /api/v1/contents/ai-edit/rewrite` | 重写优化 |
| `POST /api/v1/contents/ai-edit/expand` | 内容扩展 |
| `POST /api/v1/contents/ai-edit/improve-tone` | 语气改善 |
| `POST /api/v1/contents/ai-edit/summarize` | 内容摘要 |
| `POST /api/v1/contents/ai-edit/translate` | 翻译 |
| `POST /api/v1/contents/ai-edit/check-banned-words` | 敏感词检测 |
| `POST /api/v1/contents/ai-edit/check-banned-words-ai` | AI 辅助敏感词检测 |
| `POST /api/v1/contents/ai-edit/suggest-tags` | 推荐标签 |
| `POST /api/v1/contents/ai-edit/generate-headlines` | 生成标题 |
| `POST /api/v1/contents/ai-edit/analyze-seo` | SEO 分析 |
| `POST /api/v1/contents/ai-edit/adapt-content` | 平台内容适配 |
| `POST /api/v1/contents/ai-edit/expand-keywords` | 关键词扩展 |
| `POST /api/v1/contents/ai-edit/analyze-viral` | 爆款分析 |
| `POST /api/v1/contents/ai-edit/parse-url` | URL 内容解析 |

### 发布与平台

| 端点 | 说明 |
|------|------|
| `GET /api/v1/platforms` | 平台列表 |
| `GET /api/v1/platforms/models` | 平台可用模型列表 |
| `POST /api/v1/publish` | 发布内容到平台 |
| `POST /api/v1/publish/batch` | 批量发布 |
| `GET /api/v1/publish/best-time` | 推荐发布时间 |
| `GET /api/v1/publish/{record_id}` | 发布记录详情 |
| `GET /api/v1/publish/content/{content_id}` | 内容的发布历史 |

### 对话引擎

| 端点 | 说明 |
|------|------|
| `POST /api/v1/conversation/chat` | 对话引擎（intent 检测 / trigger 生成） |

> **模块关系**：`src/conversation/` 模块提供 `ConversationEngine` 能力，由 `api/conversation.py` 的 `/conversation/chat` 端点暴露；`api/chat.py` 的 `/chat/*` 端点则负责对话历史的 CRUD 管理，两者各司其职。

## 环境变量

### LLM 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ZHIPU_API_KEY` | 智谱 API Key | — |
| `ZHIPU_MODEL` | 智谱模型 | `glm-4-flash` |
| `QWEN_API_KEY` | 通义千问 Key | — |
| `QWEN_MODEL` | 通义千问模型 | `qwen-plus` |
| `DEEPSEEK_API_KEY` | DeepSeek Key | — |
| `DEEPSEEK_MODEL` | DeepSeek 模型 | `deepseek-chat` |
| `OPENAI_API_KEY` | OpenAI Key | — |
| `BOCHA_API_KEY` | Bocha Search Key | — |
| `DEFAULT_LLM_PROVIDER` | 默认供应商 | `zhipu` |
| `LLM_GATEWAY_URL` | LLM Gateway URL | `http://localhost:9010/v1` |

### Per-Role 模型覆盖（可选）

| 变量 | 说明 |
|------|------|
| `MANAGER_MODEL` | ManagerAgent 专用模型 |
| `WRITING_MODEL` | WritingAgent 专用模型 |
| `IMAGE_MODEL` | ImageAgent 专用模型 |
| `SUBAGENT_MODEL` | 其他子 Agent 专用模型 |

### MCP 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MCP_ENABLED` | 启用 MCP | `true` |
| `MCP_ENABLE_SEARCH` | WebSearch Server | `true` |
| `MCP_ENABLE_BRAVE` | Brave Search | `false` |
| `MCP_ENABLE_JINA` | Jina Reader | `true` |
| `MCP_ENABLE_TAVILY` | Tavily Search | `false` |
| `MCP_ENABLE_PEXELS` | Pexels | `false` |
| `MCP_ENABLE_GITHUB` | GitHub | `false` |

### Worker 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `WORKER_MAX_RETRIES` | 任务重试次数 | `3` |
| `ORCHESTRATOR_TIMEOUT` | 编排超时秒数 | `180.0` |
| `TASK_CONCURRENCY` | 最大并发子 Agent 数 | `3` |

## 关键设计决策

| 决策 | 方案 | 权衡 |
|------|------|------|
| **LLM 出口** | 统一经过 LLM Gateway | 集中治理（限流/熔断/缓存/日志） |
| **LLM 调用路径** | ChatOpenAI → Gateway | 所有 Provider 通过 OpenAI 兼容接口 |
| **AutoRouter** | `model=auto` + `X-Route-Hint` | 成本优化路由，无需手动指定模型 |
| **编排引擎** | LangGraph StateGraph | 结构化 DAG，5 节点 |
| **实时推送** | SSE + EventBus (asyncio.Queue) | 替代轮询，事件驱动，无前端定时器 |
| **记忆系统** | 三层 MemoryManager | 跨会话上下文复用，避免 token 浪费 |
| **平台适配** | PLATFORM_PROFILES 写作 DNA | 每个平台独立 System Prompt + 写作规则 |
| **可观测性** | AgentTracer → UnifiedEsWriter | 无 OTel exporter，轻量 span 直接写 ES |
| **ES 写入** | 异步队列 + 批量刷新 + JSONL 回退 | 不阻塞主流程，ES 不可用时优雅降级 |

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v3.0.0 | 2026-06 | Review-Fix 循环（最多 2 轮）；AgentTracer 轻量追踪；UnifiedEsWriter；LLM Gateway AutoRouter |
| v2.2.0 | 2026-06 | SubAgent ReAct 循环；per-role ModelRouter；SSE EventBus |
| v2.1.0 | 2026-06 | LLM 调用统一经过 LLM Gateway；shared-logging；ES 异步写入 |
| v2.0.0 | 2026-05 | LangGraph StateGraph；MCP 集成；多层记忆系统 |
| v1.x | 2026-04 | 单 Agent 简单调用 |
