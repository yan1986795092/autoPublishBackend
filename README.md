# AutoPublish Backend

AI 内容生产与发布工具后端 API 服务，基于 FastAPI + LangGraph 多 Agent 编排引擎。

## 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| **Web 框架** | FastAPI | 异步 API + 自动文档 |
| **Agent 编排** | LangGraph | StateGraph DAG 调度 |
| **LLM 调用** | langchain_openai | 直接调用多供应商 API |
| **多供应商** | OpenAI / 智谱 / 通义千问 | 原生 base_url 兼容 |
| **数据库** | SQLAlchemy + SQLite | 内容、用户、模板存储 |
| **认证** | Python-JOSE + Passlib | JWT Token |
| **外部工具** | MCP (Model Context Protocol) | 搜索、爬虫等工具标准化 |
| **可观测性** | OpenTelemetry + Prometheus | 全链路追踪 + 指标采集 |
| **异步队列** | asyncio.Queue | 内存任务队列 |
| **实时推送** | SSE + WebSocket | 任务进度实时推送 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件填入实际配置

# 启动服务
python -m src.main
# 或
uvicorn src.main:app --reload
```

API 文档：http://localhost:8000/docs

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
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│                  Agent 编排层 (LangGraph)                               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Orchestrator                               │  │
│  │  任务拆解 + 调度中枢 + 结果合并                                │  │
│  └────────────────────────────┬─────────────────────────────────┘  │
│                               │                                       │
│  ┌────────────────────────────▼─────────────────────────────────┐  │
│  │                    LangGraph StateGraph                        │  │
│  │  classify → schedule → execute → reflect → merge             │  │
│  └────────────────────────────┬─────────────────────────────────┘  │
│                               │                                       │
│  ┌────────────────────────────▼─────────────────────────────────┐  │
│  │                    核心 Agent 团队                             │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │  │
│  │  │Manager │ │Writing │ │  SEO   │ │Review  │ │Diagram │   │  │
│  │  │Agent   │ │Agent   │ │Agent   │ │Agent   │ │Agent   │   │  │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘   │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │  │
│  │  │HotTopic│ │  Web   │ │Platform│ │  Code  │ │Translate│   │  │
│  │  │Agent   │ │Search  │ │Adapter │ │Agent   │ │Agent   │   │  │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                               │                                       │
│  ┌────────────────────────────▼─────────────────────────────────┐  │
│  │                    支撑系统 (可选)                            │  │
│  │  💬 MessageBus  │  🪞 ReflectionEngine  │  🧰 ToolRegistry  │  │
│  │  👤 HITLManager  │  🧠 ContextManager    │  💾 LLM Cache     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│              LLM 调用层 (langchain_openai)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐       │
│  │  OpenAI      │  │   智谱       │  │   通义千问            │       │
│  │  gpt-4o-mini │  │   glm-4      │  │   qwen-plus          │       │
│  │  api.openai  │  │  bigmodel.cn │  │  dashscope.aliyuncs  │       │
│  └──────────────┘  └──────────────┘  └──────────────────────┘       │
│                                                                     │
│  ModelRouter 按角色创建独立 LLM 实例，支持多角色差异化配置              │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│              MCP 工具层 (Model Context Protocol)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐ │
│  │ WebSearch    │  │ Jina Reader  │  │ Brave Search  │  │ Pexels  │ │
│  │ Server       │  │ Server       │  │ Server       │  │ Server  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Tavily      │  │ GitHub       │  │ 更多工具...  │              │
│  │ Search      │  │ Server       │  │              │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│              数据与可观测层                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ SQLite       │  │OpenTelemetry │  │Elasticsearch │              │
│  │ (本地开发)   │  │ + Prometheus │  │ (可选)       │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└───────────────────────────────────────────────────────────────────────┘
```

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
│   ├── main.py                 # FastAPI 入口 + lifespan
│   ├── config.py               # Pydantic Settings 配置管理
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
│   │   ├── ai_editor.py       # AI 编辑助手
│   │   └── skills.py           # Skill CRUD + 导入
│   ├── agent/                  # Agent 系统
│   │   ├── orchestrator.py     # 编排器 (LangGraph 包装)
│   │   ├── graph.py            # LangGraph StateGraph
│   │   ├── nodes.py            # 图节点函数
│   │   ├── edges.py            # 条件边路由
│   │   ├── state.py            # AgentGraphState
│   │   ├── factory.py          # 工厂函数
│   │   ├── api.py              # Agent REST/SSE/WS 端点
│   │   ├── schemas.py          # AgentTask / AgentResult
│   │   ├── thinking_trace.py   # 思考链路
│   │   ├── persistence.py      # 持久化
│   │   ├── cache.py            # 结果缓存
│   │   ├── retry.py            # 重试策略
│   │   ├── agents/             # Agent 基类与注册表
│   │   │   ├── base_agent.py
│   │   │   ├── profile.py
│   │   │   └── registry.py
│   │   ├── handlers/           # Agent 处理函数
│   │   │   ├── writing.py
│   │   │   ├── seo.py
│   │   │   ├── review.py
│   │   │   ├── diagram.py
│   │   │   ├── code.py
│   │   │   ├── hot_topic.py
│   │   │   ├── web_search.py
│   │   │   ├── platform_adapter.py
│   │   │   ├── translation.py
│   │   │   ├── factcheck.py
│   │   │   └── fallback.py
│   │   ├── reflection/         # 三层反思引擎
│   │   │   ├── engine.py
│   │   │   ├── self_reflection.py
│   │   │   ├── peer_review.py
│   │   │   └── meta_cognition.py
│   │   ├── communication/      # Agent 间通信
│   │   │   ├── bus.py
│   │   │   └── channel.py
│   │   ├── hitl/               # Human-in-the-Loop
│   │   │   ├── manager.py
│   │   │   └── gates.py
│   │   ├── tools/              # Agent 工具系统
│   │   │   ├── base.py
│   │   │   ├── registry.py
│   │   │   └── builtins/
│   │   └── workflow/
│   │       ├── task_queue.py   # 异步任务队列
│   │       └── worker.py       # Worker
│   ├── mcp/                    # MCP (Model Context Protocol)
│   │   ├── client.py           # MCPClient 多 Server 管理
│   │   ├── web_search_server.py
│   │   ├── jina_reader_server.py
│   │   ├── brave_search_server.py
│   │   ├── tavily_search_server.py
│   │   ├── pexels_server.py
│   │   └── github_server.py
│   ├── memory/                 # 记忆系统
│   │   ├── context_manager.py  # ContextManager
│   │   ├── context_store.py    # SharedContextStore
│   │   ├── manager.py          # MemoryManager
│   │   ├── episodic_memory.py   # 情节记忆
│   │   ├── semantic_memory.py  # 语义记忆
│   │   └── working_memory.py   # 工作记忆
│   ├── skills/                 # Skill 内置注册表
│   │   └── builtins.py         # 内置 Skill 定义
│   ├── core/                   # 核心模块
│   │   ├── llm_gateway.py     # LangChain ChatOpenAI 适配器
│   │   ├── model_router.py    # 按角色创建 LLM 实例
│   │   ├── prompt_engine.py   # Prompt 引擎
│   │   ├── content_service.py  # 内容生成服务
│   │   └── security.py        # 安全认证
│   ├── adapters/              # 平台适配器
│   │   ├── base.py
│   │   ├── xiaohongshu.py
│   │   ├── wechat.py
│   │   └── zhihu.py
│   ├── utils/
│   │   ├── evaluator.py       # 违禁词/SEO 评估
│   │   └── tokenizer.py       # Token 计数
│   ├── prompts/               # Prompt 模板
│   │   ├── engine.py
│   │   ├── registry.py
│   │   └── builtins/
│   ├── observability/         # 可观测性
│   │   ├── tracing.py
│   │   ├── metrics.py
│   │   └── es_client.py
│   └── schemas/
├── tests/
├── scripts/
├── docs/
├── .env.example
└── requirements.txt
```

## LLM 调用

使用 `langchain_openai.ChatOpenAI` 直接调用各供应商 API：

```python
# ModelRouter 按角色创建 ChatOpenAI 实例
from src.core.model_router import ModelRouter

router = ModelRouter()
llm = router.get_chat_model("WritingAgent")
response = await llm.ainvoke(messages)
```

### 多供应商支持

```python
# OpenAI
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1

# 智谱 GLM
ZHIPU_API_KEY=xxx
ZHIPU_MODEL=glm-4-flash
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# 通义千问 QWEN
QWEN_API_KEY=xxx
QWEN_MODEL=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

默认优先级：`ZHIPU_API_KEY` → `QWEN_API_KEY` → `OPENAI_API_KEY`

### 按角色差异化配置

支持为不同 Agent 角色指定不同模型：

| 角色 | 说明 | 默认模型 |
|------|------|----------|
| `manager` | 任务拆解与规划 | 智谱 glm-4-flash |
| `writing` | 内容生成 | 智谱 glm-4-flash |
| `image` | 图片生成 | 通义 qwen-plus |
| `subagent` | 子 Agent (Review, SEO 等) | 智谱 glm-4-flash |

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
```

## 关键设计决策

| 决策 | 方案 | 权衡 |
|------|------|------|
| **LLM 调用** | langchain_openai 直接调用 | 轻量透明，无额外代理开销 |
| **编排引擎** | LangGraph StateGraph | 结构化 DAG，可预测性强 |
| **任务队列** | asyncio.Queue (内存) | 零依赖，单进程零开销 |
| **工具集成** | MCP (Model Context Protocol) | 标准化热插拔协议 |
| **模型路由** | ModelRouter 按角色创建 | 支持差异化配置和缓存 |
| **上下文管理** | 三层记忆 (Working/Episodic/Semantic) | 平衡实时性与长期学习 |
| **可选 Phase** | 特性开关 (enable_*) | 按需启用，避免复杂度 |

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v2.0.0 | 2026-06 | 重构 LLM 调用层，移除 Model Gateway，使用 langchain_openai 直接调用 |
| v1.x | 早期 | 初始架构，litellm + Model Gateway |
