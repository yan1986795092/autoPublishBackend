# AutoPublish — 项目规格说明

> **最后更新**: 2026-06-25
> **架构版本**: v3.0.0

## 项目概述

AI 驱动的内容生产与多平台发布工具。支持从选题、写作、SEO 优化、审核到多平台适配的全流程自动化。

核心能力：
- **多平台内容适配**：10 个发布平台 + 5 种内容类型的深度写作 DNA
- **Multi-Agent 编排**：基于 LangGraph StateGraph 的 DAG 任务调度，支持并行执行、依赖管理与自动重试
- **实时流式体验**：SSE + EventBus 驱动的实时任务进度推送
- **记忆系统**：工作记忆、语义记忆、情景记忆三层上下文管理
- **MCP 工具集成**：WebSearch / BraveSearch / Tavily / Jina Reader / Pexels / GitHub

## 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React 18 + Vite)               │
│  /  /content  /content/:id  /settings  /operations        │
└───────────────────────┬───────────────────────────────────┘
                        │ HTTP / SSE (WebSocket)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend API (FastAPI, :9000)                  │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Agent Orchestrator (LangGraph)          │  │
│  │  understand_intent → plan_tasks → schedule_execute     │  │
│  │  → execute_task_batch ──────────────────────────────►│  │
│  │       ↑                   ↓                         │  │
│  │       └────── should_continue_or_finish ◄───────────┘  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  Agent Handlers: Writing | Review | Fix | SEO | Diagram    │
│                   | Code | WebSearch | URLCrawler | HotTopic│
│  Memory System: WorkingMemory | SemanticMemory | Episodic  │
│  MCP Client: Search | Brave | Jina | Tavily | Pexels | GH  │
│  Logging: structlog + UnifiedEsWriter (async batch)        │
│  Observability: AgentTracer → UnifiedEsWriter (ES)          │
└───────────────────────┬───────────────────────────────────┘
                        │ model=auto / deepseek-chat
                        ▼
┌─────────────────────────────────────────────────────────────┐
│            LLM Gateway (:9010, OpenAI-compatible)           │
│  POST /v1/chat/completions  │  GET /v1/models            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Plugin Chain: RateLimiter → Cache → Observability   │  │
│  │  Router: AutoRouter (model=auto) + ExactMatch       │  │
│  │  Adapters: OpenAI | Zhipu | Qwen | DeepSeek         │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────┬───────────────────────────────────┘
                        │ httpx
                        ▼
              External LLM Providers
```

## 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| **Web 框架** | FastAPI | 异步 API，自动文档 |
| **Agent 编排** | LangGraph | StateGraph DAG，5 节点编排 |
| **LLM 调用** | LangChain (ChatOpenAI) | 统一指向 LLM Gateway |
| **LLM 网关** | LLM Gateway (:9010) | AutoRouter 智能路由 + 限流/熔断/缓存 |
| **多供应商** | OpenAI / 智谱 / 通义千问 / DeepSeek | 通过 Gateway 统一封装 |
| **数据库** | SQLAlchemy + SQLite/PostgreSQL | 内容、用户、任务存储 |
| **认证** | JWT (Python-JOSE + Passlib) | Token 刷新机制 |
| **外部工具** | MCP (Model Context Protocol) | langchain-mcp-adapters |
| **可观测性** | structlog + UnifiedEsWriter | 三层索引：traces / agent-events / logs |
| **实时推送** | SSE + EventBus (asyncio.Queue) | 替代轮询，事件驱动 |
| **前端** | React 18 + Vite + Tailwind CSS + Framer Motion + Radix UI + Lucide |

## 项目结构

```
autoGitub/
├── backend/
│   └── src/
│       ├── main.py                    # FastAPI 入口，lifespan，11 个异常处理器
│       ├── config.py                  # Pydantic Settings（含 per-role LLM 配置）
│       ├── database.py                # SQLAlchemy 引擎 + 会话管理
│       ├── migrations.py              # DB 迁移注册
│       │
│       ├── api/                      # REST API 路由
│       │   ├── auth.py               # JWT 认证
│       │   ├── contents.py          # 内容 CRUD / 生成 / 日历 / 版本
│       │   ├── publish.py            # 多平台发布
│       │   ├── platforms.py          # 平台元数据
│       │   ├── presets.py            # 预设模板
│       │   ├── ai_editor.py         # AI 编辑器
│       │   ├── personas.py           # 角色预设
│       │   ├── skills.py            # Skill CRUD / 导入导出
│       │   ├── chat.py              # 对话历史（会话/消息 CRUD）
│       │   ├── conversation.py       # 对话引擎（intent 检测 / trigger）
│       │   ├── admin.py              # 管理路由
│       │   └── agent/
│       │       └── api.py            # Agent 运行 / 计划 / 重利用 / SSE 流
│       │
│       ├── agent/                    # Agent 系统
│       │   ├── orchestrator.py        # Orchestrator（LangGraph StateGraph 包装）
│       │   ├── graph.py              # build_agent_graph() — 5 节点 DAG
│       │   ├── nodes.py              # 全部节点函数（意图理解/规划/调度/执行/合并）
│       │   ├── state.py              # AgentGraphState TypedDict
│       │   ├── factory.py           # create_orchestrator()
│       │   ├── schemas.py           # AgentResult / AgentTask / ExecutionPlan
│       │   ├── result.py            # agent_response / error_response 构造器
│       │   ├── registry.py           # AgentCard 注册表 + CATEGORY_DEPENDENCY_RULES
│       │   ├── stage_map.py         # Agent → 阶段标签/图标映射
│       │   ├── platform_registry.py  # 平台元数据（10 平台）
│       │   ├── platform_profile.py   # 平台写作 DNA（15 种内容形态完整配置）
│       │   ├── cache.py             # agent_cache / llm_cache
│       │   ├── persistence.py       # save_agent_task / save_subtask
│       │   ├── agent_utils.py       # extract_output / UNDERSTANDING_SYSTEM_PROMPT 等
│       │   ├── output_filter.py     # strip_meta_dump / content_quality_check
│       │   ├── workflow/
│       │   │   ├── worker.py         # AgentWorker（异步任务执行 + 重试策略）
│       │   │   └── task_queue.py    # TaskQueue / EventBus / AsyncTask
│       │   └── handlers/            # Agent Handler 实现
│       │       ├── __init__.py      # HANDLER_MAP 导出
│       │       ├── writing.py        # WritingAgent — 多风格/多平台内容生成
│       │       ├── review.py         # ReviewAgent — 质量审核
│       │       ├── fix.py           # FixAgent — 基于审核反馈优化
│       │       ├── seo.py           # SEOAgent — SEO 评分/关键词优化
│       │       ├── diagram.py       # DiagramAgent — Mermaid 图表生成
│       │       ├── code.py          # CodeAgent — 代码片段生成
│       │       ├── web_search.py    # WebSearch / URLCrawler
│       │       ├── hot_topic.py     # HotTopicAgent — 热点抓取分析
│       │       ├── tool_utils.py    # call_llm（带流式 chunk）
│       │       └── fallback.py      # handle_fallback
│       │
│       ├── memory/                   # 记忆系统
│       │   ├── manager.py           # MemoryManager（统一入口）
│       │   ├── working_memory.py    # WorkingMemory（会话内上下文）
│       │   ├── semantic_memory.py    # SemanticMemory（向量语义存储）
│       │   ├── episodic_memory.py    # EpisodicMemory（情景记忆）
│       │   ├── context_store.py     # ContextStore（agent-scoped 读写）
│       │   ├── context_builder.py   # ContextBuilder（多源上下文聚合）
│       │   ├── context_manager.py   # ContextManager（上下文生命周期）
│       │   ├── consolidator.py     # MemoryConsolidator（后台巩固）
│       │   ├── chat_history.py     # ChatHistory
│       │   ├── agent_scoped_view.py # Agent 视角的上下文读写
│       │   ├── schemas.py           # 记忆数据结构
│       │   └── tools.py            # run_memory_tool / MEMORY_TOOLS
│       │
│       ├── core/                     # 核心模块
│       │   ├── model_router.py      # ModelRouter（自动路由 + 指定模型）
│       │   ├── security.py           # JWT / Passlib / decode_token
│       │   ├── exceptions.py        # 统一异常类 + build_error_response
│       │   ├── request_guards.py   # ensure_task_owner / validate_public_http_url
│       │   └── migrations.py        # DB migration_manager
│       │
│       ├── services/                 # 业务服务层
│       │   ├── content_service.py   # 内容生成/批量/SSE 流式
│       │   ├── content_parser_service.py
│       │   └── ai_editor_service.py
│       │
│       ├── prompts/                  # Prompt 管理
│       │   ├── builder.py          # PromptBuilder（模块化构建）
│       │   ├── registry.py         # PromptRegistry（模板注册）
│       │   ├── engine.py           # PromptEngine
│       │   ├── registry_loader.py
│       │   ├── registry_types.py
│       │   └── builtins/          # 内置 Prompt 模板
│       │       ├── system.py       # system.base / system.writing
│       │       ├── writing.py
│       │       ├── review.py
│       │       ├── code.py
│       │       ├── diagram.py
│       │       └── manager.py
│       │
│       ├── mcp/                      # MCP (Model Context Protocol)
│       │   ├── client.py           # MCPServerConfig + MCPClient 多服务器管理
│       │   ├── web_search_server.py
│       │   ├── brave_search_server.py
│       │   ├── tavily_search_server.py
│       │   ├── jina_reader_server.py
│       │   ├── pexels_server.py
│       │   └── github_server.py
│       │
│       ├── conversation/             # 对话引擎
│       │   ├── engine.py           # ConversationEngine
│       │   ├── intent_detector.py  # IntentDetector（trigger 生成）
│       │   └── schemas.py
│       │
│       ├── logging_shared/          # 统一日志基础设施
│       │   ├── config.py           # structlog 初始化
│       │   ├── unified_writer.py   # UnifiedEsWriter（异步批量写 ES + JSONL 回退）
│       │   ├── middleware.py      # LoggingMiddleware（trace_id）
│       │   ├── fields.py          # 日志字段常量
│       │   ├── console.py         # Rich Console 输出
│       │   ├── agent_emitter.py   # AgentEventEmitter
│       │   └── trace_context.py
│       │
│       ├── observability/           # 可观测性
│       │   ├── agent_tracer.py    # AgentTracer（轻量 span → UnifiedEsWriter）
│       │   ├── metrics.py         # Prometheus 指标（up_gauge / record_agent_run）
│       │   ├── tracing.py         # OTel SDK 初始化（无 exporter，span 数据走 AgentTracer）
│       │   └── es_client.py      # ES 客户端
│       │
│       └── tools/                  # 工具函数
│           ├── vector_search.py    # FAISS 向量搜索
│           ├── code_runner.py     # 代码执行沙箱
│           ├── tokenizer.py        # Token 计数
│           └── langchain/         # LangChain 集成
│               └── crawler_tool.py
│
├── llm-gateway/                    # LLM Gateway 服务（:9010）
│   └── gateway/
│       ├── main.py                 # FastAPI 入口，lifespan
│       ├── api/
│       │   ├── routes.py          # POST /v1/chat/completions（核心）
│       │   ├── schemas.py        # Pydantic 请求/响应模型
│       │   └── dependencies.py    # build_adapters / build_router 等
│       ├── core/
│       │   ├── router.py         # AutoRouter（model=auto 路由策略）
│       │   ├── context.py        # RequestContext
│       │   └── exceptions.py    # GatewayError / CircuitBreakerOpenError
│       ├── adapters/              # Provider 适配器
│       │   ├── base.py
│       │   └── openai.py         # OpenAI 兼容适配器
│       ├── plugins/               # 中间件插件
│       │   ├── base.py
│       │   ├── rate_limiter.py   # 滑动窗口限流
│       │   ├── cache.py         # Redis 响应缓存
│       │   ├── circuit_breaker.py # 状态机熔断
│       │   └── observability.py  # 请求日志
│       └── logging_shared/       # 共享日志（backend 同款）
│           ├── config.py / middleware.py / unified_writer.py
│           ├── es_writer.py / fields.py / console.py
│
├── frontend/                      # React 前端
│   └── src/
│       ├── App.tsx               # 路由：HomePage / ContentManagement /
│       │                         #   ContentEdit / Settings / Operations
│       ├── pages/                # 页面组件
│       ├── components/
│       │   ├── agent/           # Agent 工作区（AdvancedPanel / StageTimeline）
│       │   ├── content/          # 内容管理（ContentCalendar / PublishDialog）
│       │   ├── preview/          # 多平台预览（15 种内容形态的预览组件）
│       │   ├── layout/          # AppShell / TopNav / ConversationHistory
│       │   ├── operations/       # 运维中心（CostTab / SystemTab / PublishingTab）
│       │   ├── settings/         # 设置页面
│       │   └── ui/              # Radix UI 基础组件
│       ├── stores/              # Zustand 状态管理
│       ├── hooks/               # React hooks
│       └── api/                  # 前端 API 客户端
│
├── infra/                         # 基础设施配置
├── docs/                          # 项目文档
└── scripts/                       # 脚本工具
```

## LangGraph Agent 编排

### 节点图

```
understand_intent  ──►  plan_tasks  ──►  schedule_and_execute
                                           │ (conditional)
                     ┌────────────────────┴────────────────────┐
                     │ should_continue_or_finish               │
                     ▼                                         ▼
              execute_task_batch  ──────────────────────────────►│
                     ↑                                         │
                     │ should_continue_or_finish               │
                     └──────────────────────────────────────────┘
                                                                   │
                                                                   ▼
                                                            merge_and_finish  ──► END
```

### 5 个核心节点

| 节点 | 职责 | 关键逻辑 |
|------|------|---------|
| `understand_intent` | LLM 意图分析 | 提取 intent / slots / platform / language；规则兜底 |
| `plan_tasks` | 任务拆解 | LLM 生成 / 规则兜底；注入理解槽位 + 前端参数；依赖链修正 |
| `schedule_and_execute` | DAG 调度 | 拓扑排序；依赖解析；死锁检测 |
| `execute_task_batch` | 并行执行 | asyncio.gather 并发；semaphore 限流；Review-Fix 循环（最多 2 轮） |
| `merge_and_finish` | 结果合并 | FixAgent > ReviewFix > WritingAgent 优先级；异步持久化到 DB |

### 条件边路由

- `should_continue_or_finish` → `"execute"`：有待执行任务
- `should_continue_or_finish` → `"done"`：所有任务已完成或死锁

## Agent 团队

| Agent | Handler | 功能 | 默认模型策略 |
|-------|---------|------|------------|
| WritingAgent | `handle_writing` | 多风格内容创作 | auto-route (creative) |
| ReviewAgent | `handle_review` | 质量审核、违禁词检查 | auto-route (analysis) |
| FixAgent | `handle_fix` | 基于审核反馈优化内容 | auto-route (creative) |
| SEOAgent | `handle_seo` | SEO 评分、关键词优化 | auto-route (creative) |
| DiagramAgent | `handle_diagram` | Mermaid 图表生成 | auto-route (creative) |
| CodeAgent | `handle_code` | 代码片段生成 | 指定模型 (deepseek-chat) |
| WebSearch | `handle_web_search` | MCP 网络搜索 | auto-route (simple) |
| URLCrawler | `handle_url_crawl` | 页面内容抓取 | — |
| HotTopicAgent | `handle_hot_topic` | 热点抓取分析 | — |
| ManagerAgent | (orchestrator 内) | 任务拆解与规划 | 指定模型 (deepseek-chat) |

## 平台与内容形态支持（15 种）

`PLATFORM_PROFILES`（`platform_profile.py`）定义每个发布平台 / 内容类型的完整写作画像，分三类：

| 分类 | 数量 | 说明 |
|------|------|------|
| 发布平台 | 6 | 小红书、微信公众号、微博、知乎、抖音、B站 |
| 内容创作 | 5 | 产品文案、品牌故事、广告文案、小说故事、剧本脚本 |
| 专业写作 | 4 | 技术文档、教程指南、SEO文章、邮件营销 |

| 发布平台 / 内容类型 | 代码 | 分类 | 写作规则 | SEO 改写 |
|------|------|------|---------|---------|
| 小红书 | `xiaohongshu` | 社交媒体 | 口语化、emoji、300-1000字 | 是 |
| 微信公众号 | `wechat` | 社交媒体 | 深度文章、2000-5000字 | 否 |
| 微博 | `weibo` | 社交媒体 | 观点鲜明、50-500字 | 否 |
| 抖音 | `douyin` | 社交媒体 | 口播脚本、hook 结构 | 否 |
| B站 | `bilibili` | 社交媒体 | 脚本+弹幕文化、800-3000字 | 否 |
| 知乎 | `zhihu` | 社交媒体 | 专业深度问答 | 否 |
| 产品文案 | `copywriting` | 内容创作 | 营销转化、CTA | 是 |
| 品牌故事 | `brand-story` | 内容创作 | 叙事感、真诚有温度 | 否 |
| 广告文案 | `ad-copy` | 内容创作 | 效果广告、30-300字 | 是 |
| 小说故事 | `fiction` | 内容创作 | 虚构叙事、1000-10000字 | 否 |
| 剧本脚本 | `script` | 内容创作 | 分镜格式、对白潜台词 | 否 |
| 技术文档 | `tech-doc` | 专业写作 | 中性技术语气、代码示例 | 是 |
| 教程指南 | `tutorial` | 专业写作 | 步骤教学、常见误区 | 是 |
| SEO文章 | `seo` | 专业写作 | E-E-A-T 权威性 | 是 |
| 邮件营销 | `email` | 专业写作 | 打开率优化、CTA | 否 |

## API 端点

### Agent

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

### 内容

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

### 对话

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

### 发布

| 端点 | 说明 |
|------|------|
| `POST /api/v1/contents/{id}/publish` | 发布到平台 |
| `POST /api/v1/publish/batch` | 批量发布 |

## 记忆系统

三层架构（`MemoryManager` 统一入口）：

| 层级 | 存储 | 生命周期 | 用途 |
|------|------|---------|------|
| **WorkingMemory** | 进程内存 | 会话内 | 当前对话上下文、token 预算追踪 |
| **SemanticMemory** | SQLite/向量 | 跨会话 | 用户偏好、内容片段的语义检索 |
| **EpisodicMemory** | SQLite | 跨会话 | 情景记忆（Agent 输入/输出摘要） |

关键流程：
- `build_agent_context()`：Agent 执行前，从语义记忆检索相关上下文注入 prompt
- `remember()`：Agent 执行后，将高质量产出选择性沉淀到语义记忆
- `consolidate()`：后台任务，从最近完成的 AgentTask 中提取内容向量化存储

## 可观测性

### 三层数据 → UnifiedEsWriter

| 数据类型 | 索引 | 数据来源 | 用途 |
|---------|------|---------|------|
| **Traces** | `traces-YYYY.MM.DD` | AgentTracer（轻量 span） | 链路追踪（JSON 结构写入 ES，查询 ES 拼出调用链） |
| **Agent Events** | `agent-events-YYYY.MM.DD` | AgentEventEmitter | 推理过程、工具调用、成本 |
| **Logs** | `logs-YYYY.MM.DD` | structlog | 问题排查、审计 |

### ES 写入策略

- **异步批量写入**：队列缓冲 + 5s 刷新间隔或 batch=100 触发
- **JSONL 回退**：ES 不可用时写入 `logs/traces-events-fallback.jsonl`
- **非阻塞**：写入错误不中断主流程

### Prometheus 指标

- `up_gauge` / `info_gauge`：进程存活状态
- `record_agent_run()`：每次 Agent 执行的 status / skill / duration
- `record_agent_task()`：每个任务的类型 / 状态 / 耗时

## LLM Gateway

架构职责：

| 职责 | 实现 |
|------|------|
| **请求转发** | `/v1/chat/completions` OpenAI 兼容接口 |
| **AutoRouter** | `model=auto` + `X-Route-Hint`（creative/simple/analysis）按成本选 provider |
| **精确路由** | 指定模型名 → 直接匹配对应 adapter |
| **限流** | 滑动窗口 RPM/TPM 限流（per API key） |
| **缓存** | Redis 响应缓存（按请求 hash 去重） |
| **熔断** | 状态机熔断（单 Provider 故障不影响其他） |
| **Provider Fallback** | 主 Provider 失败 → 自动切换到下一个可用 Provider |
| **日志** | ObservabilityPlugin 记录所有请求详情 |

## 启动方式

```bash
# LLM Gateway（必须先启动）
cd llm-gateway
pip install -r requirements.txt
cp .env.example .env    # 编辑 .env 填入实际配置
uvicorn gateway.main:app --port 9010

# Backend
cd backend
pip install -r requirements.txt
cp .env.example .env   # 编辑 .env
uvicorn src.main:app --reload --port 9000

# Frontend
cd frontend && npm install && npm run dev
```

## 环境变量（核心）

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
| **Agent 工厂** | create_orchestrator() → ModelRouter | 各 Agent 按类型获取独立 LLM 实例 |

## 架构演进历史

| 版本 | 日期 | 关键变更 |
|------|------|---------|
| v3.0.0 | 2026-06 | Review-Fix 循环（最多 2 轮）；AgentTracer 轻量追踪；UnifiedEsWriter；LLM Gateway AutoRouter |
| v2.2.0 | 2026-06 | SubAgent ReAct 循环；per-role ModelRouter；SSE EventBus |
| v2.1.0 | 2026-06 | LLM 调用统一经过 LLM Gateway；shared-logging；ES 异步写入 |
| v2.0.0 | 2026-05 | LangGraph StateGraph；MCP 集成；多层记忆系统 |
| v1.x | 2026-04 | 单 Agent 简单调用 |
