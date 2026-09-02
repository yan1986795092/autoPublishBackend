# 墨掌柜 Backend

> AI 驱动的内容生产 API —— 输入一个主题，输出一篇可在多平台发布的高质量内容。

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2%2B-orange)]()

---

## 快速开始

**60 秒内跑通第一个请求：**

```bash
# 1. 安装依赖
cd backend && pip install -r requirements.txt

# 2. 配置环境变量（至少一个 LLM API Key）
cp .env.example .env
# 编辑 .env，填入 ZHIPU_API_KEY 或 OPENAI_API_KEY 等

# 3. 启动服务
uvicorn src.main:app --reload --port 9000

# 4. 请求预览计划
curl -X POST http://localhost:9000/api/v1/agent/plan \
  -H "Content-Type: application/json" \
  -d '{"prompt": "写一篇露营装备推荐的小红书笔记"}'
```

> **前置依赖**: 需要先启动 LLM Gateway（平级目录 `llm-gateway/`）或配置直连 Provider。
> 详见 [.env.example](.env.example)。

---

## 核心概念

| 概念 | 一句话 |
|------|--------|
| **Agent** | 一个"数字编辑"，各司其职（写作、审核、搜索...） |
| **ManagerAgent** | 主编，理解你想干啥，然后派活 |
| **Orchestrator** | 编排引擎（LangGraph），像流水线一样调度 Agent |
| **SSE 流式** | 实时看到 Agent 的思考过程，不干等 |
| **PlatformProfile** | 每个平台的"写作 DNA"——小红书该怎么写、知乎该怎么写 |
| **Skill** | 可复用的"内容生产 SOP"——写热点文需要 5 个步骤，存成一个 Skill |
| **Review-Fix 循环** | 自动审稿→修改→再审（最多 2 轮），保证质量 |
| **Task Queue** | 异步任务队列，提交后可以离开，回来时已完成 |

### Agent 团队

| Agent | 像谁 | 做什么 |
|-------|------|--------|
| **ManagerAgent** | 主编 | 理解意图、拆解任务、派活 |
| **WritingAgent** | 主笔 | 写文章、写文案、写脚本 |
| **ReviewAgent** | 审稿编辑 | 检查质量、违禁词，给出修改意见 |
| **FixAgent** | 责任编辑 | 按审稿意见改文章 |
| **WebSearch** | 研究员 | 联网搜资料 |
| **HotTopicAgent** | 分析师 | 分析热点趋势 |
| **DiagramAgent** | 美术 | 画图表（Mermaid） |
| **CodeAgent** | 程序员 | 写代码、写技术文档 |
| **SEOAgent** | SEO专家 | SEO优化分析 |
| **FactCheckAgent** | 事实核查 | 核查内容准确性 |

### 内容支持（15 种形态）

```
社交平台: 小红书 | 公众号 | 微博 | 抖音 | B站 | 知乎
内容创作: 产品文案 | 品牌故事 | 广告文案 | 小说 | 脚本
专业写作: 技术文档 | 教程 | SEO 文章 | 邮件营销
```

---

## API 速查

### Agent 执行

| 模式 | 端点 | 说明 |
|------|------|------|
| 同步 | `POST /agent/run` | 等结果返回 |
| SSE 流式 v3 | `POST /agent/stream/v3` | 实时看到思考过程（推荐） |
| SSE 流式 v2 | `POST /agent/stream/v2` | 兼容旧版，2026-09-30 下线 |
| SSE 流式 v1 | `POST /agent/stream` | deprecated，2026-09-30 下线 |
| Observability SSE | `GET /agent/obs/stream/{task_id}` | 内部状态/全量 events（dev panel 用） |
| 异步 | `POST /agent/async/submit` | 提交就走，回来收草稿 |
| WebSocket | `WS /agent/ws/{id}` | 双向实时进度 |

### 内容 & 编辑

```bash
# 内容 CRUD
GET    /api/v1/contents              # 列表
POST   /api/v1/contents/generate     # AI 生成

# AI 编辑器（14 个功能）
POST   /api/v1/contents/ai-edit/rewrite       # 重写
POST   /api/v1/contents/ai-edit/translate     # 翻译
POST   /api/v1/contents/ai-edit/analyze-seo   # SEO 分析
POST   /api/v1/contents/ai-edit/analyze-viral # 爆款分析
POST   /api/v1/contents/ai-edit/suggest-tags  # 标签推荐
POST   /api/v1/contents/ai-edit/generate-headlines  # 标题生成
```

> 完整端点列表见 [SPEC.md](./SPEC.md)

---

## SSE 前端契约（spec 2026-07-02）

`/agent/stream/v3` 是当前推荐的流式接口，采用 **Parts 模型**：

### v3 事件类型

| 事件 | 说明 |
|------|------|
| `message_start` | 消息开始，设置 streamingMessageId |
| `plan` | 规划预览，追加 PlanPart |
| `stage:started` | Agent 阶段开始，追加 ToolCallPart |
| `stage:completed` | Agent 阶段完成，状态流转到 done |
| `tool_input_delta` | 工具输入更新（部分 JSON 合并） |
| `tool_result` | 工具结果输出 |
| `reasoning_delta` | 推理过程流式片段 |
| `text_delta` | 正文流式片段 |
| `artifact` | 产物追加（标题/标签/正文） |
| `message_end` | 消息结束，关闭流式状态 |
| `error` | 错误信息 |

### v3 vs v2 差异

- v3 使用 Parts 模型，前端通过 `eventToPartOp` 转换器处理事件
- v3 支持更细粒度的流式更新（reasoning_delta、tool_input_delta）
- v2/v1 仍可用，但将于 2026-09-30 下线

---

## 架构一览

```
frontend/               # React 前端 (平级目录)
  │ HTTP / SSE / WebSocket
  ▼
backend/ (:9000)        # 后端服务
  ├── LangGraph 编排     # 5 节点 DAG，ManagerAgent 调度
  ├── Agent Handlers    # 每个 Handler 是一个专业"编辑"
  ├── 三层记忆系统       # Working / Semantic / Episodic
  ├── 对话引擎           # 意图检测 + Chat CRUD
  └── 异步任务队列       # 提交→执行→DLQ→重试
  │ ChatOpenAI 协议
  ▼
llm-gateway/ (:9010)    # LLM 网关 (平级目录)
  │ OpenAI / 智谱 / 通义千问 / DeepSeek
```

---

## 后端结构

```
backend/
├── src/
│   ├── main.py                 # FastAPI 入口
│   ├── config.py               # 配置管理
│   ├── database.py             # 数据库连接
│   ├── migrations.py           # 数据迁移
│   ├── api/                    # REST 路由
│   │   ├── agent.py           # Agent 执行接口
│   │   ├── contents.py        # 内容管理
│   │   ├── chat.py            # 对话接口
│   │   ├── conversation.py    # 对话历史
│   │   ├── auth.py            # 认证授权
│   │   ├── publish.py         # 发布接口
│   │   ├── platforms.py       # 平台配置
│   │   ├── skills.py          # 技能模板
│   │   ├── presets.py         # 预设管理
│   │   └── admin.py           # 管理后台
│   ├── agent/                  # Agent 编排系统
│   │   ├── orchestrator.py    # 编排器
│   │   ├── graph.py           # LangGraph 图定义
│   │   ├── nodes.py           # 5 个节点函数
│   │   ├── handlers/          # Agent 实现
│   │   │   ├── writing.py     # WritingAgent
│   │   │   ├── review.py      # ReviewAgent
│   │   │   ├── fix.py         # FixAgent
│   │   │   ├── web_search.py   # WebSearch
│   │   │   ├── seo.py         # SEOAgent
│   │   │   ├── diagram.py     # DiagramAgent
│   │   │   ├── code.py        # CodeAgent
│   │   │   └── ...
│   │   ├── workflow/          # 异步队列
│   │   │   ├── task_queue.py  # 任务队列
│   │   │   └── worker.py      # Worker
│   │   ├── registry.py        # AgentCard 注册表
│   │   ├── platform_profile.py # 平台配置
│   │   └── state.py          # 图状态
│   ├── conversation/           # 对话引擎
│   │   ├── engine.py          # 对话处理
│   │   └── intent_detector.py # 意图检测
│   ├── core/                   # 核心模块
│   │   ├── model_router.py    # 模型路由
│   │   ├── security.py        # 安全认证
│   │   └── tenant.py         # 租户隔离
│   ├── memory/                 # 记忆系统
│   ├── logging_shared/         # 统一日志
│   └── observability/          # 可观测性
├── tests/
├── .env.example
└── requirements.txt
```

---

## 环境变量

**LLM Provider**（至少配一个）：

| 变量 | Provider | 默认模型 |
|------|----------|---------|
| `ZHIPU_API_KEY` | 智谱 | `glm-4-flash` |
| `OPENAI_API_KEY` | OpenAI | `gpt-3.5-turbo` |
| `QWEN_API_KEY` | 通义千问 | `qwen-plus` |
| `DEEPSEEK_API_KEY` | DeepSeek | `deepseek-chat` |

**MCP 工具**（按需启用）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `MCP_ENABLE_SEARCH` | `true` | 联网搜索 |
| `MCP_ENABLE_BRAVE` | `false` | Brave Search (需 `BRAVE_API_KEY`) |
| `MCP_ENABLE_JINA` | `true` | 网页内容抓取 |
| `MCP_ENABLE_PEXELS` | `false` | 图片搜索 (需 `PEXELS_API_KEY`) |

> 全部环境变量见 [.env.example](.env.example)
