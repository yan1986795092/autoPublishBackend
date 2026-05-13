# AutoPublish Backend

AI 内容生产与发布工具后端 API 服务，基于 FastAPI + LangGraph Multi-Agent 编排引擎。

## 技术栈

- **FastAPI** — Web 框架
- **LangGraph** — Multi-Agent StateGraph 编排
- **LangChain** — LLM 调用封装
- **SQLAlchemy** — ORM（SQLite 开发环境）
- **Python-JOSE + Passlib** — JWT 认证 + 密码加密
- **Pydantic** — 数据验证
- **httpx** — HTTP 客户端（搜索、热点抓取）
- **sse-starlette** — SSE 流式推送
- **Jinja2** — Prompt 模板引擎

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
┌─────────────────────────────────────────────┐
│              API Layer (FastAPI)              │
│  /agent/*  /contents/*  /publish/*           │
└────────────┬────────────────────┬────────────┘
             │                    │
┌────────────▼──────┐  ┌─────────▼──────────┐
│  Orchestrator     │  │  Content Services   │
│  (LangGraph)      │  │  (CRUD + Generate)  │
└────────────┬──────┘  └────────────────────┘
             │
┌────────────▼────────────────────────────────┐
│          Agent Handlers (HANDLER_MAP)        │
│  Writing │ SEO │ Review │ WebSearch         │
│  Diagram │ Image │ HotTopic │ Code          │
│  PlatformAdapter │ Translation │ Fallback   │
└────────────┬────────────────────────────────┘
             │
┌────────────▼────────────────────────────────┐
│         LLM Gateway / ModelRouter            │
│  OpenAI │ Qwen │ Zhipu                      │
└─────────────────────────────────────────────┘
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
              ▼
    merge_and_finish → 结果合并 + 持久化
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

### 已注册 Agent（10 个）

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
| **WebSearch** | 实时网络搜索 | SerpAPI / DuckDuckGo |
| **PlatformAdapter** | 多平台格式转换 | 平台适配器 |

### 任务拆解策略

| 复杂度 | 触发条件 | 拆解方式 | 步骤数 |
|--------|---------|---------|-------|
| simple | 无关键词、短文本 | 模板 | 2 |
| medium | 含搜索/平台/优化关键词 | LLM + 模板兜底 | 4 |
| complex | 含图表/配图/多平台等 | LLM + 验证回退 | 6 |

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
│   │   └── ai_editor.py        # AI 编辑助手
│   ├── agent/                  # Agent 系统
│   │   ├── orchestrator.py     # 编排器 + 任务拆解
│   │   ├── graph.py            # LangGraph StateGraph
│   │   ├── nodes.py            # 图节点函数
│   │   ├── state.py            # AgentGraphState
│   │   ├── factory.py          # 工厂函数
│   │   ├── api.py              # Agent REST/SSE/WS 端点
│   │   ├── schemas.py          # AgentTask / AgentResult
│   │   ├── thinking_trace.py   # 思考链路记录
│   │   ├── persistence.py      # 持久化
│   │   ├── agent_utils.py      # 共享常量（违禁词/模板）
│   │   ├── handlers/           # Agent 处理函数
│   │   │   ├── writing.py / seo.py / review.py
│   │   │   ├── diagram.py / image.py / hot_topic.py
│   │   │   ├── code.py / web_search.py
│   │   │   ├── platform_adapter.py / translation.py
│   │   │   ├── fallback.py     # 兜底处理
│   │   │   └── __init__.py     # HANDLER_MAP
│   │   └── workflow/
│   │       ├── task_queue.py   # 异步任务队列
│   │       └── worker.py       # Worker
│   ├── services/
│   │   ├── content_service.py  # 内容生成 + 管理
│   │   └── ai_editor_service.py # AI 编辑器服务
│   ├── adapters/               # 平台适配器
│   │   ├── base.py             # 抽象基类
│   │   ├── xiaohongshu.py
│   │   ├── wechat.py
│   │   └── zhihu.py
│   ├── core/
│   │   ├── llm_gateway.py      # LLM 网关
│   │   ├── model_router.py     # 模型路由
│   │   └── prompt_engine.py    # Prompt 引擎
│   ├── utils/
│   │   └── evaluator.py        # 违禁词/SEO 评估
│   ├── schemas/                # Pydantic Schema
│   └── prompts/
│       └── agent_prompts.py    # Prompt 模板
├── tests/                      # 单元测试（75 passed）
├── scripts/                    # 工具脚本
├── .env.example
├── Dockerfile
└── requirements.txt
```

## LLM 提供商配置

支持三个提供商，通过环境变量切换：

| 提供商 | 环境变量 | 默认模型 |
|--------|---------|---------|
| OpenAI | `OPENAI_API_KEY` | gpt-3.5-turbo |
| 通义千问 | `DASHSCOPE_API_KEY` | qwen-plus |
| 智谱 | `ZHIPUAI_API_KEY` | glm-4 |

各 Agent 可通过 `ModelRouter` 分配独立的模型实例，实现模型分级调度。

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

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行 Agent 单元测试
pytest tests/agents/test_base_agent.py -v
```

## 关键设计决策

| 决策 | 方案 | 权衡 |
|------|------|------|
| 编排引擎 | LangGraph StateGraph | 结构化 DAG vs 自由 ReAct，可预测性优先 |
| 任务队列 | 内存 asyncio.Queue | 零依赖，但重启丢失。适合单进程 |
| 计划传递 | SSE 预计算 → Worker 复用 | 消除计划不一致，但增加一次 LLM 调用 |
| 数据库隔离 | 批量生成创建独立 Session | 防竞态，但增加连接开销 |
| 违禁词检查 | 正则 + 子串双系统 | Agent 审核精确，编辑器检查灵活 |
