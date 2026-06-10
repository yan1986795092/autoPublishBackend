# AutoPublish — 项目规格说明

> **最后更新**: 2026-06-10
> **架构版本**: v2.2.0

## 项目概述

AI 驱动的内容生产与多平台发布工具。支持从选题、写作、SEO 优化、审核到多平台适配的全流程自动化。基于 FastAPI + LangGraph 多 Agent 编排引擎，所有 LLM 调用统一经过 LLM Gateway 服务。

## 系统架构图

```
Frontend (React 18 + Vite)
    ↓ HTTP / SSE
Backend API (FastAPI, :9000)
    ↓ ModelRouter.get_chat_model(agent_type) → ChatOpenAI
LLM Gateway (:9010)
    ↓ httpx
External LLM Providers (OpenAI / 智谱 / 通义千问)
```

### 后端模块分层

| 层级 | 模块 | 职责 |
|------|------|------|
| **API 路由** | `api/` | REST 端点，JWT 认证 |
| **Agent 编排** | `agent/` | LangGraph StateGraph，任务调度 |
| **Agent 执行** | `agent/handlers/` | WritingAgent / SEOAgent / ReviewAgent 等 |
| **Agent 通信** | `agent/communication/` | MessageBus P2P 通信 |
| **Agent 反思** | `agent/reflection/` | Self / Peer / Meta 三层反思 |
| **ReAct 子系统** | `agent/react/` | SubAgent 独立 ReAct 循环 |
| **HITL** | `agent/hitl/` | Human-in-the-Loop 决策门控 |
| **工具系统** | `agent/tools/` | ToolRegistry，MCP 集成 |
| **记忆系统** | `memory/` | SharedContextStore，ContextManager |
| **可观测性** | `observability_v2/` | OTel + ES + structlog 统一 |
| **LLM 层** | `core/` | ModelRouter（per-role → Gateway） |

## 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| **Web 框架** | FastAPI | 异步 API + 自动文档 |
| **Agent 编排** | LangGraph | StateGraph DAG 调度 |
| **LLM 调用** | LangChain + LLM Gateway | ChatOpenAI → Gateway → 多供应商 |
| **多供应商** | OpenAI / 智谱 / 通义千问 | Gateway 统一封装 |
| **数据库** | SQLAlchemy + SQLite | 内容、用户、模板存储 |
| **认证** | Python-JOSE + Passlib | JWT Token |
| **外部工具** | MCP (Model Context Protocol) | 搜索、爬虫等工具标准化 |
| **可观测性** | OpenTelemetry + OTel Collector + Prometheus + Elasticsearch | 全链路追踪 + 指标采集 + 结构化日志 |
| **日志** | structlog | 中文结构化 JSON，Console + ES 异步写入 |

## 项目结构

```
autoGitub/
├── backend/                   # FastAPI 后端
│   ├── src/
│   │   ├── main.py           # FastAPI 入口 + lifespan
│   │   ├── config.py         # Pydantic Settings
│   │   ├── database.py       # SQLAlchemy 引擎
│   │   ├── models.py         # ORM 模型
│   │   ├── logging_shared/   # 统一日志基础设施
│   │   │   ├── fields.py     # 日志字段常量
│   │   │   ├── config.py     # structlog 统一配置
│   │   │   └── es_writer.py  # ES 异步批量写入器
│   │   ├── api/              # REST API 路由
│   │   ├── agent/           # Agent 系统 (LangGraph)
│   │   │   ├── orchestrator.py
│   │   │   ├── graph.py
│   │   │   ├── nodes.py
│   │   │   ├── edges.py
│   │   │   ├── state.py
│   │   │   ├── factory.py
│   │   │   ├── schemas.py
│   │   │   ├── agents/       # BaseAgent + Registry + Profile
│   │   │   ├── handlers/     # Writing / SEO / Review 等
│   │   │   ├── reflection/    # 三层反思引擎
│   │   │   ├── communication/  # MessageBus P2P
│   │   │   ├── hitl/         # Human-in-the-Loop
│   │   │   ├── react/        # SubAgent ReAct 循环
│   │   │   ├── tools/         # ToolRegistry
│   │   │   └── workflow/       # 异步任务队列
│   │   ├── memory/           # 记忆系统
│   │   ├── core/             # 核心模块
│   │   │   ├── model_router.py   # per-role LLM 路由
│   │   │   ├── prompt_engine.py
│   │   │   └── security.py
│   │   ├── adapters/         # 平台适配器
│   │   ├── observability_v2/  # 统一可观测性
│   │   └── utils/
│   ├── tests/
│   └── docs/
│
├── llm-gateway/              # LLM Gateway 服务
│   ├── gateway/
│   │   ├── main.py
│   │   ├── api/routes.py     # OpenAI 兼容 API
│   │   ├── core/
│   │   │   ├── router.py     # AutoRouter (已精简为透明代理)
│   │   │   ├── context.py
│   │   │   └── exceptions.py
│   │   ├── plugins/
│   │   │   ├── rate_limiter.py
│   │   │   ├── cache.py
│   │   │   ├── circuit_breaker.py
│   │   │   └── observability.py
│   │   ├── adapters/
│   │   │   ├── base.py
│   │   │   └── openai.py
│   │   ├── logging_shared/    # 共享日志副本
│   │   └── config/settings.py
│   └── README.md
│
├── frontend/                 # React 前端
├── infra/
│   ├── otel/               # OTel Collector 配置
│   └── grafana/             # Grafana 配置
├── docs/                    # 文档
└── logs/                   # 日志目录
```

## 启动方式

```bash
# LLM Gateway（必须先启动）
cd llm-gateway
uvicorn gateway.main:app --port 9010

# 后端
cd backend
pip install -r requirements.txt
cp .env.example .env    # 编辑 .env 填入实际配置
uvicorn src.main:app --reload --port 9000

# 前端
cd frontend && npm install && npm run dev
```

## LLM Gateway

LLM Gateway 是系统的统一 LLM 出口，架构职责：

| 职责 | 说明 |
|------|------|
| **请求转发** | 将 OpenAI 兼容请求转发到对应 Provider |
| **限流** | 滑动窗口算法，按 key 限 RPM/TPM |
| **缓存** | Redis 存储响应，按请求 hash 去重 |
| **熔断** | 状态机熔断，单个 Provider 故障不影响其他 |
| **日志** | ObservabilityPlugin 记录所有 LLM 请求详情到 structlog |

## LangGraph Agent 编排

### 节点图

```
classify_and_decompose  ──►  schedule_and_execute  ──►  execute_task_batch
                                ↑                         │
                                │                         ▼
                                └───── reflect_on_task_batch ──► merge_and_finish
```

### 6 个核心节点

| 节点 | 职责 |
|------|------|
| `classify_and_decompose` | LLM 意图识别 + 任务拆解，模板兜底 |
| `schedule_and_execute` | DAG 拓扑排序，ready_tasks 可执行队列 |
| `execute_task_batch` | asyncio.gather 并行执行子 Agent |
| `reflect_on_task_batch` | 质量评估(Self/Peer/Meta)，决定重试 |
| `check_hitl_gates` | HITL 人工介入门控 |
| `merge_and_finish` | 聚合所有结果，返回最终输出 |

### 可选特性（特性开关控制）

| 特性 | 开关 | 说明 |
|------|------|------|
| Agent Registry | `enable_agent_registry` | Agent 注册与工厂模式 |
| Message Bus | `enable_message_bus` | Agent 间 P2P 通信 |
| Reflection | `enable_reflection` | Self/Peer/Meta 三层反思 |
| HITL | `enable_hitl` | 人工介入决策门控 |
| SubAgent ReAct | `enable_subagent_react` | 子 Agent 独立 ReAct 循环 |

## Agent 团队

| Agent | 功能 | 默认模型 |
|-------|------|----------|
| ManagerAgent | 任务拆解与规划 | glm-4-flash |
| WritingAgent | 多风格内容创作 | glm-4-flash |
| SEOAgent | SEO 评分、关键词优化 | glm-4-flash |
| ReviewAgent | 质量审核、违禁词检查 | glm-4-flash |
| DiagramAgent | Mermaid 图表生成 | qwen-plus |
| ImageAgent | 配图生成 | - |
| HotTopicAgent | 热点抓取分析 | - |
| CodeAgent | 代码片段生成 | - |
| WebSearch | 网络搜索 (MCP) | glm-4-flash |
| PlatformAdapter | 多平台格式转换 | - |
| TranslationAgent | 多语言翻译 | - |
| FactCheckAgent | 事实核查 | glm-4-flash |

## 记忆系统

多层上下文管理，支持多轮对话和跨会话全局记忆：

| 层级 | 内容 | 可见性 |
|------|------|--------|
| **SystemLayer** | user_request, project_manifest, global_memory, thinking_trace | 所有 Agent 可读 |
| **TaskLayer** | writing_content, review_feedback, seo_suggestions, creative_anchors | 按权限读写 |
| **AgentLayer** | internal_reasoning, confidence_score | Agent 私有 |

### 创意锚点（Creative Anchors）

创作任务中的核心元素永不压缩：
- `story_outline` — 故事大纲
- `character_profiles` — 角色设定
- `world_setting` — 世界观设定
- `style_guide` — 风格指南
- `plot_milestones` — 关键情节点

### 项目阶段（ProjectStage）

| 阶段 | 说明 |
|------|------|
| `CONCEPTUALIZATION` | 构思期：大量发散，保留所有创意 |
| `OUTLINING` | 大纲期：锚点建立，结构成型 |
| `DRAFTING` | 初稿期：保留近期内容块 + 全部锚点 |
| `REVISION` | 修改期：保留修改段落 + 相关反馈 |
| `POLISHING` | 润色期：保留风格指南和待润色文本 |
| `COMPLETED` | 完成：固化锚点，关闭会话 |
| `ABANDONED` | 废弃：保存状态，复盘记录 |

## 可观测性

### 三层数据模型

| 层级 | 索引 | 数据来源 | 用途 |
|------|------|----------|------|
| HTTP 链路追踪 | `traces-*` | OTelMiddleware | Jaeger UI 链路图 |
| Agent 业务事件 | `agent-events-*` | AgentEventEmitter | 推理过程、工具调用、成本 |
| 应用日志 | `logs-*` | structlog | 问题排查、审计 |

### 关键可配置项

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `LOG_JSON_FORMAT` | JSON 格式 | `false` |
| `ELASTICSEARCH_URL` | ES 地址 | `http://localhost:9200` |

## API 端点

### Agent

| 端点 | 说明 |
|------|------|
| `POST /api/v1/agent/run` | 同步执行 Agent 编排 |
| `POST /api/v1/agent/plan` | 预览执行计划 |
| `POST /api/v1/agent/stream` | SSE 流式执行 |
| `POST /api/v1/agent/async/submit` | 提交异步任务 |
| `GET /api/v1/agent/async/status/{id}` | 查询任务状态 |
| `WS /api/v1/agent/ws/{id}` | WebSocket 实时进度 |
| `POST /api/v1/agent/decision/{id}` | 提交用户决策 |
| `GET /api/v1/agent/dlq` | 死信队列 |
| `GET /api/v1/agent/metrics` | Agent 监控指标 |

### 内容

| 端点 | 说明 |
|------|------|
| `GET/POST /api/v1/contents` | 内容列表 / 新建 |
| `GET/PUT/DELETE /api/v1/contents/{id}` | 内容 CRUD |
| `POST /api/v1/contents/{id}/generate` | AI 生成内容 |
| `POST /api/v1/contents/{id}/generate-stream` | 流式生成 |
| `POST /api/v1/contents/batch-generate` | 批量生成 |
| `GET /api/v1/contents/{id}/versions` | 版本历史 |
| `GET /api/v1/contents/calendar` | 内容日历 |

### Skills

| 端点 | 说明 |
|------|------|
| `GET/POST /api/v1/skills` | Skill 列表 / 创建 |
| `GET/PUT/DELETE /api/v1/skills/{id}` | Skill CRUD |
| `GET /api/v1/skills/builtins` | 内置 Skill |
| `POST /api/v1/skills/import/url` | 从 URL 导入 |
| `POST /api/v1/skills/import/file` | 从文件导入 |

### 发布

| 端点 | 说明 |
|------|------|
| `POST /api/v1/contents/{id}/publish` | 发布到平台 |
| `POST /api/v1/publish/batch` | 批量发布 |
| `GET /api/v1/platforms` | 平台列表 |
| `POST /api/v1/ai/generate` | AI 内容生成 |
| `POST /api/v1/ai/edit` | AI 编辑优化 |
| `POST /api/v1/ai/translate` | AI 翻译 |
| `POST /api/v1/ai/check-banned-words` | 违禁词检查 |
| `POST /api/v1/ai/analyze-seo` | SEO 分析 |

## 环境变量

### LLM 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ZHIPU_API_KEY` | 智谱 API Key | - |
| `ZHIPU_MODEL` | 智谱模型 | `glm-4-flash` |
| `QWEN_API_KEY` | 通义千问 Key | - |
| `QWEN_MODEL` | 通义千问模型 | `qwen-plus` |
| `OPENAI_API_KEY` | OpenAI Key | - |
| `OPENAI_MODEL` | OpenAI 模型 | `gpt-4o-mini` |
| `GATEWAY_URL` | LLM Gateway URL | `http://localhost:9010` |

### MCP 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MCP_ENABLED` | 启用 MCP | `true` |
| `MCP_ENABLE_SEARCH` | WebSearch Server | `true` |
| `MCP_ENABLE_JINA` | Jina Reader | `true` |
| `MCP_ENABLE_BRAVE` | Brave Search | `false` |
| `MCP_ENABLE_TAVILY` | Tavily Search | `false` |
| `MCP_ENABLE_PEXELS` | Pexels | `false` |
| `MCP_ENABLE_GITHUB` | GitHub | `false` |

### 可选功能

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ENABLE_REFLECTION` | 反思引擎 | `false` |
| `ENABLE_MESSAGE_BUS` | Agent 通信 | `false` |
| `ENABLE_HITL` | 人工介入 | `false` |
| `ENABLE_SUBAGENT_REACT` | SubAgent ReAct | `false` |

## 关键设计决策

| 决策 | 方案 | 权衡 |
|------|------|------|
| **LLM 出口** | 统一经过 LLM Gateway | 集中治理（限流/熔断/缓存/日志），后端轻量 |
| **LLM 调用路径** | 统一走 ChatOpenAI → Gateway | 删除 LLMGateway httpx 客户端，LangChain 路径唯一 |
| **Per-Role 路由** | ModelRouter + get_role_config() | 不同 Agent 可用不同模型，无需改代码 |
| **编排引擎** | LangGraph StateGraph | 结构化 DAG，条件边路由，6 个节点 |
| **SubAgent ReAct** | 可选，基于 BaseAgent 叠加 run_with_react() | 不改变现有 Agent，渐进升级 |
| **多轮对话** | SharedContextStore + 权限分片 | Agent 仅看到有权限的上下文 |
| **可观测性** | OTelMiddleware → OTel Collector → ES | 统一追踪，三层数据，Jaeger UI |
| **日志** | shared-logging + ES 异步批量写入 | trace_id 跨服务串联，写入不阻塞 |
| **可选 Phase** | 特性开关 | 按需启用，避免复杂度 |

## 架构演进历史

| 版本 | 日期 | 关键变更 |
|------|------|----------|
| v2.2.0 | 2026-06 | SubAgent ReAct 循环；per-role ModelRouter；OTelMiddleware 统一可观测性 |
| v2.1.0 | 2026-06 | LLM 调用统一经过 LLM Gateway；shared-logging；ES 异步写入 |
| v2.0.0 | 2026-05 | LangGraph StateGraph；MCP 集成；多层记忆系统 |
| v1.x | 2026-04 | 单 Agent 简单调用 |
