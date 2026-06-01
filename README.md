# AutoPublish Backend

AI 内容生产与发布工具后端 API 服务，基于 FastAPI + LangGraph Multi-Agent 编排引擎。

## 技术栈

- **FastAPI** — Web 框架
- **LangGraph** — Multi-Agent StateGraph 编排
- **LangChain** — LLM 调用封装
- **litellm** — 多供应商统一调用（OpenAI / 智谱 / 通义千问 / 100+ 供应商）
- **SQLAlchemy** — ORM（SQLite 开发环境）
- **Python-JOSE + Passlib** — JWT 认证 + 密码加密
- **Pydantic** — 数据验证
- **httpx** — HTTP 客户端（搜索、热点抓取）
- **sse-starlette** — SSE 流式推送
- **Jinja2** — Prompt 模板引擎
- **MCP (Model Context Protocol)** — 外部工具标准化集成
- **OpenTelemetry + Prometheus** — 全链路追踪与可观测性

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件填入实际配置

# 启动服务
python run.py
```

API 文档：http://localhost:8000/docs

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                                  │
│  /agent/*  /contents/*  /publish/*  /api/v1/skills/*              │
└────────────┬────────────────────┬─────────────────┬──────────────────┘
             │                    │                 │
┌────────────▼──────┐  ┌────────▼────────┐  ┌─▼─────────────────┐
│  Orchestrator     │  │  Content Services│  │  Skills API        │
│  (LangGraph)      │  │  (CRUD+Generate)│  │  CRUD+Import+URL   │
└────────────┬──────┘  └─────────────────┘  └────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────────┐
│                     Agent System (Multi-Agent)                          │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────────┐    │
│  │  ToolRegistry │  │ ContextManager │  │ ReflectionEngine  │    │
│  │  注册+绑定    │  │ 混合路由+压缩   │  │ Self+Peer+Meta   │    │
│  └──────┬───────┘  └───────┬───────┘  └────────────────────┘    │
│         │                    │                                       │
│  ┌──────▼───────┐  ┌───────▼────────┐  ┌────────────────────┐    │
│  │ Handler层     │  │ MemoryManager  │  │ HITLManager       │    │
│  │ Writing/SEO   │  │ Working+Epis   │  │ Human-in-Loop     │    │
│  │ Review/Diagram│  │ +Semantic       │  │ 审批门控          │    │
│  └───────────────┘  └────────────────┘  └────────────────────┘    │
└────────────┬───────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────────┐
│                    MCP Layer (Model Context Protocol)                    │
│  MCPClient — 配置驱动的多 Server 管理（stdio / streamable-http）        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ WebSearch │ Brave Search │ Jina Reader │ Tavily │ Pexels │     │
│  │ Server    │ Server       │ Server      │ Search │ Server │     │
│  │ GitHub    │              │             │ Server │        │     │
│  │ Server    │             │             │        │        │     │
│  └────────────────────────────────────────────────────────────────┘ │
└────────────┬───────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────────┐
│              Model Gateway（生产级，OpenAI 兼容）                       │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────┐  ┌────────┐ │
│  │Security MW    │  │RateLimiter│  │CircuitBrk   │  │ Router │ │
│  │来源校验+白名单│  │滑动窗口RPM │  │被动+主动     │  │三维加权│ │
│  └──────────────┘  └────────────┘  └─────────────┘  └────────┘ │
│  ┌──────────────┐  ┌────────────────────────────────────────┐     │
│  │ CostTracker   │  │ Tracer (OTel + Prometheus)             │     │
│  │ 异步批量写入  │  │ 100%采样 + 全链路追踪                  │     │
│  └──────────────┘  └────────────────────────────────────────┘     │
│                            │                                         │
│  litellm — 统一调用 OpenAI / 智谱 / 通义千问 / 100+ 供应商         │
└──────────────────────────────────────────────────────────────────────┘
```

### Multi-Agent 编排流程

```
用户请求 → Orchestrator.run()
              │
              ▼
    classify_and_decompose → 模板/LLM 拆解
              │
              ▼
    find_ready_tasks → DAG 依赖分析
              │
              ▼
    execute_task_batch → asyncio.gather 并行
         │
         │  每个 task 走完整链路：
         │  Skill Prompt → 记忆注入 → 工具描述 → Token Budget
         │       │
         │       ▼
         │  GatewaySession (内嵌网关)
         │  ├─ EnhancedRouter.select()    三维加权打分选供应商
         │  ├─ MultiDimensionalRateLimiter 滑动窗口 RPM/TPM 限流
         │  ├─ CircuitBreaker             CLOSED/OPEN/HALF_OPEN 熔断
         │  ├─ LiteLLMClient.completion()  OTel span + litellm 调用
         │  ├─ LatencyTracker.record()    P95 延迟反馈回路
         │  └─ CostTracker.record()       成本落库 gateway_cost.db
         │
         ▼
    reflect_on_task_batch → Self/Peer/Meta 三层反思
              │
              ▼
    merge_and_finish → 结果合并 + 持久化 + 记忆巩固
```

支持**预计算计划传递**：SSE 端点生成的计划通过 `AsyncTask.plan` 传递给 Worker，确保用户看到的计划与 Worker 执行的计划一致。

### 异步工作流

```
Client → POST /agent/stream
           │
           ├─ ① _decompose() → 生成计划
           ├─ ② enqueue → TaskQueue (asyncio.Queue)
           ├─ ③ AgentWorker 出队执行
           │     ├─ orchestrator.run()
           │     ├─ 进度更新 → SSE push
           │     └─ 结果 → complete_task
           └─ ④ 轮询进度 → SSE 推送
                 event: plan / progress / thinking_step
                 event: decision_required / result / done
```

- **SSE** — `/agent/stream` 流式事件推送
- **WebSocket** — `/agent/ws/{task_id}` 实时进度
- **前端重连** — 指数退避（1s→30s cap），最多 5 次 + mounted 标志防僵尸

## Agent 系统

### 已注册 Agent（14 个）

| Agent | 功能 | 工具/实现 |
|-------|------|----------|
| **ManagerAgent** | 任务拆解 + 调度中枢 | LangGraph + LLM |
| **WritingAgent** | 多风格/多类型内容创作 | LLM + 内容模板 |
| **SEOAgent** | SEO 评分、关键词优化 | LLM + 评分算法 |
| **ReviewAgent** | 质量审核、违禁词检查 | LLM + 违禁词库 |
| **DiagramAgent** | Mermaid 图表生成 | LLM |
| **ImageAgent** | 配图/封面生成 | LLM |
| **HotTopicAgent** | 微博/知乎热搜抓取 | httpx 直连 |
| **CodeAgent** | 代码片段生成 | LLM |
| **WebSearch** | 实时网络搜索 | MCP: WebSearch Server |
| **PlatformAdapter** | 多平台格式转换 | 平台适配器 |
| **TranslationAgent** | 多语言翻译 | LLM |
| **FactCheckAgent** | 事实核查 | LLM + 搜索验证 |
| **KeywordToolAgent** | 关键词研究与分析 | LLM + 工具 |
| **TrendAnalysisAgent** | 趋势分析 | LLM + 数据分析 |

### MCP 工具层

| Server | 功能 | Transport | 默认启用 |
|--------|------|---------|---------|
| **WebSearch Server** | 网页/新闻搜索 + 页面抓取 | stdio | ✅ |
| **Brave Search Server** | 隐私优先搜索 | stdio | 需 `BRAVE_API_KEY` |
| **Jina Reader Server** | 网页 Markdown 化抓取 | stdio | ✅ |
| **Tavily Search Server** | AI 优化新闻/博客搜索 | stdio | 需 `TAVILY_API_KEY` |
| **Pexels Server** | 免费高清图片搜索 | stdio | 需 `PEXELS_API_KEY` |
| **GitHub Server** | 代码/Issues/PR 搜索 | stdio | 需 `GITHUB_TOKEN` |

### 任务拆解策略

| 复杂度 | 触发条件 | 拆解方式 | 步骤数 |
|--------|---------|---------|-------|
| simple | 无关键词、短文本 | 模板 | 2 |
| medium | 含搜索/平台/优化关键词 | LLM + 模板兜底 | 4 |
| complex | 含图表/配图/多平台等 | LLM + 验证回退 | 6 |

## 上下文管理与记忆系统

### 三层记忆架构

| 层级 | 存储介质 | 内容 | 检索方式 |
|------|---------|---------|---------|
| **Working Memory** | 内存（dict） | 当前会话活跃上下文 | 直接访问 |
| **Episodic Memory** | SQLite/PG | 已完成任务完整记录 | SQL 查询 |
| **Semantic Memory** | ChromaDB（向量库） | 高价值内容片段 | ANN 向量相似度检索 |

### ContextManager 核心能力

- **混合路由**：语义相似度 + 显式指令，决定新会话或继续当前会话
- **Token Budget 监控**：实时估算，80% threshold 触发压缩
- **阶段感知压缩**：不同 ProjectStage 保留不同内容（早期保留更多思考链路）

## Skills 系统

### 内置 Skill（7 个）

| ID | 名称 | 类别 | 复杂 度 |
|----|------|------|---------|
| builtin-001 | 专业博客写作 | writing | medium |
| builtin-002 | 小红书种草 | social | simple |
| builtin-003 | 热点分析报告 | analysis | complex |
| builtin-004 | 短视频脚本 | writing | medium |
| builtin-005 | 营销文案 | writing | medium |
| builtin-006 | Newsletter 邮件 | writing | simple |
| builtin-007 | 自由创作 | writing | - |

### Skill 支持的导入方式

- **URL 导入**：支持 raw GitHub/Gist URL 自动解析 SKILL.md
- **文件导入**：SKILL.md 或 JSON 格式
- **前端导入**：SKILL.md frontmatter 解析（YAML frontmatter + Markdown body）

## API 概览

### Agent 接口

| 端点 | 说明 |
|------|------|
| `POST /agent/run` | 同步执行 Agent 编排 |
| `POST /agent/plan` | 预览执行计划 |
| `POST /agent/stream` | SSE 流式执行 |
| `POST /agent/async/submit` | 提交异步任务 |
| `GET /agent/async/status/{id}` | 查询异步任务状态 |
| `GET /agent/async/list` | 任务列表 |
| `DELETE /agent/async/{id}` | 取消任务 |
| `POST /agent/decision/{id}` | 提交用户决策 |
| `POST /agent/repurpose` | 内容再利用 Pipeline |
| `WS /agent/ws/{id}` | WebSocket 实时进度 |

### 内容接口

| 端点 | 说明 |
|------|------|
| `GET/POST /contents` | 内容列表 / 新建 |
| `GET/PUT/DELETE /contents/{id}` | 内容详情 / 更新 / 删除 |
| `POST /contents/{id}/generate` | AI 生成内容 |
| `POST /contents/{id}/generate-stream` | 流式生成 |
| `POST /contents/batch-generate` | 批量生成（独立 DB session 防竞态） |
| `GET /contents/{id}/versions` | 版本历史 |
| `GET /contents/{id}/versions/diff` | 版本对比（含 content_id 归属校验） |
| `POST /contents/{id}/check-banned-words` | 违禁词检查 |
| `GET /contents/calendar` | 内容日历（按月/年过滤，已优化 N+1） |

### Skills 接口

| 端点 | 说明 |
|------|------|
| `GET /api/v1/skills` | 列出当前用户所有 Skill |
| `GET /api/v1/skills/{id}` | 获取 Skill 详情 |
| `POST /api/v1/skills` | 创建新 Skill |
| `PUT /api/v1/skills/{id}` | 更新 Skill |
| `DELETE /api/v1/skills/{id}` | 删除 Skill |
| `GET /api/v1/skills/builtins` | 列出所有内置 Skill |
| `GET /api/v1/skills/builtins/{id}` | 获取内置 Skill 详情 |
| `POST /api/v1/skills/import/url` | 从 URL 导入 Skill |
| `POST /api/v1/skills/import/file` | 从文件导入 Skill |
| `POST /api/v1/skills/import/preview` | 预览导入内容（不保存） |

### 发布与平台

| 端点 | 说明 |
|------|------|
| `POST /contents/{id}/publish` | 发布内容到平台 |
| `POST /publish/batch` | 批量多平台发布 |
| `GET /platforms` | 平台列表 |

### AI 编辑

| 端点 | 说明 |
|------|------|
| `POST /ai/generate` | AI 内容生成 |
| `POST /ai/edit` | AI 编辑/优化 |
| `POST /ai/translate` | AI 翻译 |
| `POST /ai/check-banned-words` | AI 违禁词检查 |
| `POST /ai/analyze-seo` | SEO 分析 |

### 管理

| 端点 | 说明 |
|------|------|
| `POST /auth/register` | 注册 |
| `POST /auth/login` | 登录 |
| `POST /auth/refresh` | 刷新 Token |
| `GET /auth/me` | 当前用户信息 |
| `GET/POST /templates` | Prompt 模板管理 |
| `GET/POST /presets` | 用户预设管理 |
| `GET/POST /personas` | 人设管理 |

## 项目结构

```
backend/
├── src/
│   ├── main.py                 # FastAPI 入口 + lifespan
│   ├── config.py               # Pydantic 配置管理
│   ├── database.py             # SQLAlchemy 引擎 + SessionLocal
│   ├── models.py               # ORM 模型
│   ├── api/                    # API 路由层
│   │   ├── auth.py             # JWT 认证
│   │   ├── contents.py         # 内容 CRUD + 日历
│   │   ├── publish.py          # 发布管理
│   │   ├── platforms.py        # 平台管理
│   │   ├── templates.py        # Prompt 模板
│   │   ├── presets.py          # 用户预设
│   │   ├── personas.py         # 人设管理
│   │   ├── ai_editor.py        # AI 编辑助手
│   │   └── skills.py          # Skill CRUD + 导入 + 解析
│   ├── agent/                  # Agent 系统
│   │   ├── orchestrator.py     # 编排器 + 任务拆解
│   │   ├── graph.py            # LangGraph StateGraph
│   │   ├── nodes.py            # 图节点函数
│   │   ├── edges.py            # 条件边路由函数
│   │   ├── state.py            # AgentGraphState
│   │   ├── factory.py          # 工厂函数
│   │   ├── api.py              # Agent REST/SSE/WS 端点
│   │   ├── schemas.py          # AgentTask / AgentResult
│   │   ├── thinking_trace.py   # 思考链路记录
│   │   ├── persistence.py      # 持久化
│   │   ├── agent_utils.py      # 共享常量（违禁词/模板）
│   │   ├── cache.py            # 结果缓存
│   │   ├── retry.py            # 重试策略
│   │   ├── agents/             # Agent 基类与注册表
│   │   │   ├── base_agent.py
│   │   │   ├── profile.py
│   │   │   └── registry.py
│   │   ├── handlers/           # Agent 处理函数
│   │   │   ├── writing.py / seo.py / review.py
│   │   │   ├── diagram.py / image.py / hot_topic.py
│   │   │   ├── code.py / web_search.py
│   │   │   ├── platform_adapter.py / translation.py
│   │   │   ├── factcheck.py / keyword_tool.py / trend_analysis.py
│   │   │   ├── fallback.py
│   │   │   └── __init__.py     # HANDLER_MAP
│   │   ├── reflection/          # 三层反思引擎
│   │   │   ├── engine.py
│   │   │   ├── self_reflection.py
│   │   │   ├── peer_review.py
│   │   │   ├── meta_cognition.py
│   │   │   └── models.py
│   │   ├── communication/       # Agent 间通信
│   │   │   ├── bus.py
│   │   │   ├── channel.py
│   │   │   └── models.py
│   │   ├── hitl/               # Human-in-the-Loop
│   │   │   ├── manager.py
│   │   │   └── gates.py
│   │   ├── tools/              # Agent 工具系统
│   │   │   ├── base.py        # AgentTool / AgentToolMeta
│   │   │   ├── registry.py    # ToolRegistry / bind_agent_tools
│   │   │   ├── binding.py
│   │   │   └── builtins/       # 内置工具集
│   │   │       ├── memory_tools.py
│   │   │       ├── reflection_tools.py
│   │   │       ├── hitl_tools.py
│   │   │       └── communication_tools.py
│   │   └── workflow/
│   │       ├── task_queue.py   # 异步任务队列
│   │       ├── worker.py       # Worker
│   │       └── redis_queue.py  # Redis 队列（可选）
│   ├── mcp/                     # MCP（Model Context Protocol）
│   │   ├── client.py           # MCPClient 多 Server 管理
│   │   ├── web_search_server.py
│   │   ├── brave_search_server.py
│   │   ├── jina_reader_server.py
│   │   ├── tavily_search_server.py
│   │   ├── pexels_server.py
│   │   └── github_server.py
│   ├── memory/                  # 记忆系统
│   │   ├── context_manager.py  # ContextManager（混合路由+压缩）
│   │   ├── context_store.py    # SharedContextStore
│   │   ├── manager.py          # MemoryManager（上下文构建）
│   │   ├── episodic_memory.py   # 情节记忆
│   │   ├── semantic_memory.py  # 语义记忆（ChromaDB）
│   │   ├── working_memory.py   # 工作记忆
│   │   ├── consolidator.py     # 记忆巩固（外部触发）
│   │   ├── agent_scoped_view.py
│   │   └── schemas.py
│   ├── skills/                 # Skill 内置注册表
│   │   └── builtins.py         # 7 个内置 Skill 定义
│   ├── core/
│   │   ├── llm_gateway.py      # LLM 网关（Legacy）
│   │   ├── model_router.py     # 模型路由
│   │   ├── prompt_engine.py   # Prompt 引擎
│   │   ├── security.py         # 安全工具
│   │   ├── logging.py          # structlog 配置
│   │   ├── exceptions.py      # 自定义异常
│   │   └── migrations.py
│   ├── core/gateway/          # 模型网关（已接入 Agent）
│   │   ├── config.py           # 网关配置（路由权重/限流/价格表）
│   │   ├── server.py          # FastAPI 应用入口
│   │   ├── litellm_client.py  # litellm 封装 + callbacks
│   │   ├── tracing.py         # OTel + structlog 配置
│   │   ├── api/
│   │   │   ├── chat.py        # POST /v1/chat/completions
│   │   │   ├── models.py      # GET  /v1/models
│   │   │   └── stats.py        # GET  /gateway/stats
│   │   └── core/
│   │       ├── container.py    # GatewayContainer 单例管理器
│   │       ├── session.py      # GatewaySession (LangChain 兼容)
│   │       ├── router.py       # EnhancedRouter 三维加权打分
│   │       ├── circuit_breaker.py  # CircuitBreaker 混合熔断
│   │       ├── rate_limiter.py    # SlidingWindowRateLimiter 限流
│   │       ├── cost_tracker.py    # CostTracker 异步批量写入
│   │       └── security.py        # SecurityMiddleware 来源校验
│   ├── services/
│   │   ├── content_service.py   # 内容生成 + 管理
│   │   └── ai_editor_service.py # AI 编辑器服务
│   ├── adapters/               # 平台适配器
│   │   ├── base.py             # 抽象基类
│   │   ├── xiaohongshu.py
│   │   ├── wechat.py
│   │   └── zhihu.py
│   ├── utils/
│   │   └── evaluator.py        # 违禁词/SEO 评估
│   ├── schemas/                # Pydantic Schema
│   └── prompts/
│       └── builtin/            # 内置 Prompt 模板
│           ├── system.py
│           ├── manager.py
│           ├── writing.py
│           ├── review.py
│           ├── diagram.py
│           ├── code.py
│           └── reflection.py
├── tests/                      # 单元测试
├── scripts/                    # 工具脚本
├── docs/                        # 设计文档
├── .env.example
├── Dockerfile
└── requirements.txt
```

## 模型网关（已接入 Agent 系统）

网关通过 litellm 统一封装所有供应商，提供生产级治理能力。**每次 Agent LLM 调用都经过完整网关管线**，不再绕过：

| 组件 | 功能 |
|------|------|
| **GatewaySession** | LangChain 兼容包装器（ainvoke/astream/bind_tools），handler 零改动接入 |
| **GatewayContainer** | 单例管理器，共享 CircuitBreaker / RateLimiter / CostTracker / LatencyTracker |
| **Enhanced Router** | 三维加权打分（成本 0.4 + 静态延迟 0.3 + P95 实际延迟反馈 0.3）|
| **Rate Limiter** | 滑动窗口 RPM/TPM 限流（按供应商 + 按来源应用） |
| **Circuit Breaker** | 被动熔断（连续失败/错误率）+ 主动健康检查 + 批量半开探测 |
| **Cost Tracker** | 异步队列 + SQLite 批量写入 (`gateway_cost.db`) |
| **Latency Tracker** | 60s 滑动窗口 P95 延迟，实时反馈到路由打分 |

**调用链路**：
```
handler → call_llm() → GatewaySession.ainvoke/astream
         → EnhancedRouter.select() → RateLimiter.check()
         → LiteLLMClient.completion() → litellm.acompletion()
         → LatencyTracker.record() → CostTracker.record()
```

与旧版 `LLMGateway`（直接 ChatOpenAI 调用）的区别：
- 旧版：`ModelRouter → ChatOpenAI → 直连供应商 API`（限流/熔断/路由全无效）
- 新版：`ModelRouter → GatewaySession → 完整 Gateway 管线 → 供应商 API`

## 违禁词系统

两套独立实现：

| 系统 | 位置 | 用途 | 匹配方式 |
|------|------|------|---------|
| 通用词库 | `utils/evaluator.py` | ReviewAgent 审核 | 正则（精确+模式） |
| 按平台词库 | `services/content_service.py` | 编辑器实时检查 | 子串匹配 + 建议替换 |

`agent_utils.py` 和 `prompts/agent_prompts.py` 均导入通用词库，避免重复维护。

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DATABASE_URL` | 数据库连接 | `sqlite:///./autopublish.db` |
| `SECRET_KEY` | JWT 密钥 | - |
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `DASHSCOPE_API_KEY` | 通义千问 API Key | - |
| `ZHIPUAI_API_KEY` | 智谱 API Key | - |
| `SERPAPI_API_KEY` | SerpAPI 搜索 Key | - |
| `XHS_COOKIE` | 小红书凭证（可选） | - |
| `WECHAT_APP_ID` | 公众号 App ID（可选） | - |
| `WECHAT_APP_SECRET` | 公众号 Secret（可选） | - |
| `ZHIHU_COOKIE` | 知乎凭证（可选） | - |
| `MCP_ENABLE_SEARCH` | 启用 WebSearch Server | `true` |
| `MCP_ENABLE_BRAVE` | 启用 Brave Search Server | `false` |
| `MCP_ENABLE_JINA` | 启用 Jina Reader Server | `true` |
| `MCP_ENABLE_TAVILY` | 启用 Tavily Search Server | `false` |
| `MCP_ENABLE_PEXELS` | 启用 Pexels Server | `false` |
| `MCP_ENABLE_GITHUB` | 启用 GitHub Server | `false` |
| `GATEWAY_MODE` | 网关模式（`gateway` / `legacy`）| `legacy` |

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行 Agent 单元测试
pytest tests/agents/test_base_agent.py -v

# 运行网关冒烟测试
pytest tests/core/gateway/test_smoke.py -v
```

## 关键设计决策

| 决策 | 方案 | 权衡 |
|------|------|------|
| 编排引擎 | LangGraph StateGraph | 结构化 DAG vs 自由 ReAct，可预测性优先 |
| 任务队列 | 内存 asyncio.Queue | 零依赖，但重启丢失。适合单进程 |
| 计划传递 | SSE 预计算 → Worker 复用 | 消除计划不一致，但增加一次 LLM 调用 |
| 数据库隔离 | 批量生成创建独立 Session | 防竞态，但增加连接开销 |
| 违禁词检查 | 正则 + 子串双系统 | Agent 审核精确，编辑器检查灵活 |
| 工具集成 | MCP (Model Context Protocol) | 标准化协议，热插拔，新增工具不改代码 |
| 模型调用 | litellm 统一封装 | 支持 100+ 供应商，治理逻辑叠加 |
| 网关接入 | GatewaySession 内嵌 Agent 管线 | handler 零改动；路由/限流/熔断/成本对每次 LLM 调用生效 |
| 上下文压缩 | 阶段感知分层压缩 | 不同阶段保留不同内容，早期保留更多思考链路 |
