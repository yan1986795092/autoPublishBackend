# AutoPublish Backend

> 🤖 AI 驱动的内容生产 API —— 输入一个主题，输出一篇可在多平台发布的高质量内容。

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

## 你能用它做什么

### 一句话生成内容

```bash
curl -N -X POST http://localhost:9000/api/v1/agent/stream/v2 \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "推荐 5 个提高效率的 VS Code 插件",
    "context": {
      "platform": "zhihu",
      "tone": "professional",
      "audience": "开发者"
    }
  }'
# → SSE 流式返回: 思考分析 → 搜索资料 → 写作 → 审核 → 优化 → 完成
```

> **端点选择**：
> - `POST /agent/stream/v2` — **推荐**。前端契约收敛（spec 2026-07-01），实时不下发内部状态。
> - `POST /agent/stream` — 已标 deprecated，2026-09-30 下线（详见 [docs/streaming-frontend-contract.md](./docs/streaming-frontend-contract.md)）。

### 追热点写文章

提交一个异步任务，离开页面，回来时草稿已就绪：

```bash
TASK_ID=$(curl -s -X POST http://localhost:9000/api/v1/agent/async/submit \
  -H "Content-Type: application/json" \
  -d '{"prompt": "追今天的 AI 热点，写一篇公众号文章"}' \
  | jq -r '.task_id')

# 随时查看进度
curl http://localhost:9000/api/v1/agent/async/status/$TASK_ID
```

### 内容再利用

把公众号长文一键改写成小红书笔记：

```bash
curl -X POST http://localhost:9000/api/v1/agent/repurpose \
  -H "Content-Type: application/json" \
  -d '{
    "content": "（已发布的公众号正文...）",
    "target_platforms": ["xiaohongshu", "weibo"]
  }'
```

### AI 编辑器（单步操作）

改写、扩写、翻译、SEO 优化——不需要完整编排：

```bash
curl -X POST http://localhost:9000/api/v1/contents/ai-edit/rewrite \
  -H "Content-Type: application/json" \
  -d '{"content": "原文章...", "tone": "humorous"}'
```

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

### 内容支持（15 种形态）

```
社交平台: 小红书 | 公众号 | 微博 | 抖音 | B站 | 知乎
内容创作: 产品文案 | 品牌故事 | 广告文案 | 小说 | 脚本
专业写作: 技术文档 | 教程 | SEO 文章 | 邮件营销
```

每种形态定义了独立的写作规则、字数范围、语气标签和 System Prompt——系统自动适配，无需手动切换 Prompt。

---

## 架构一览

```
frontend/               # React 前端 (平级目录)
  │ HTTP / SSE / WebSocket
  ▼
backend/ (:9000)        # 你在这里
  ├── LangGraph 编排     # 5 节点 DAG，ManagerAgent 调度
  ├── 8 个 Agent Handler # 每个 Handler 是一个专业"编辑"
  ├── 三层记忆系统       # 工作记忆 / 语义记忆 / 情景记忆
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
│   ├── api/                    # REST 路由
│   ├── agent/                  # Agent 编排系统
│   │   ├── orchestrator.py     # 编排器
│   │   ├── graph.py            # LangGraph 图定义
│   │   ├── nodes.py            # 5 个节点函数
│   │   ├── handlers/           # Agent 实现 (writing/review/fix/search/...)
│   │   ├── registry.py         # AgentCard 注册表
│   │   └── workflow/           # 异步队列 + EventBus
│   ├── memory/                 # 三层记忆系统
│   ├── core/                   # 核心 (模型路由/认证/安全/异常)
│   ├── mcp/                    # MCP 工具客户端
│   ├── prompts/                # Prompt 模板管理
│   ├── logging_shared/         # 统一日志
│   └── observability/          # 追踪 + 指标
├── tests/
├── .env.example
└── requirements.txt
```

> 完整目录树及每文件的职责说明，见 [SPEC.md](./SPEC.md) 附录。

---

## API 速查

### Agent 执行（5 种模式 + 1 个观测轨）

| 模式 | 端点 | curl 示例 |
|------|------|-----------|
| 同步 | `POST /agent/run` | 等结果返回 |
| **SSE 流式 v2** | `POST /agent/stream/v2` | 实时看到思考过程 ✅ **推荐**（业务轨，spec 2026-07-01） |
| SSE 流式 v1 | `POST /agent/stream` | ⚠️ deprecated，2026-09-30 下线 |
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
```

> 完整端点列表见 [SPEC.md](./SPEC.md#12-appendix-api-reference)

---

## 🎯 SSE 前端契约（spec 2026-07-01）

2026-07-01 起，`/agent/stream` 的实时 SSE 流做了**分轨收敛**：

| 端点 | Audience | 实时下发事件 | `result` 事件 schema | 适用场景 |
|------|----------|-------------|---------------------|----------|
| `POST /agent/stream/v2` | 业务轨（前端） | `plan / stage / progress / llm_chunk / thinking_step / reasoning / result / done / error` | **收敛**（不含运行时缓冲 / 不含全量 events） | ✅ 前端产品（推荐） |
| `POST /agent/stream` | 业务轨（前端，兼容） | **全部**事件（含 `thinking_step / reasoning`） | **全量 dump**（含 `agent_events.events[]`） | ⚠️ deprecated，2026-09-30 下线 |
| `GET /agent/obs/stream/{task_id}` | Observability | 内部状态 + 全量 result dump | 全量 | dev panel / 后端调试 |

### 业务轨 `result` 收敛 schema（v2）

```jsonc
// SSE 下发（v2）
{
  "task_id": "...",
  "status": "completed",
  "agent_type": "WritingAgent",
  "output": {
    "full_text": "...",
    "title": "...",
    "platform": "xiaohongshu",
    "content_type": "blog",
    "draft_content": "...",     // 多轮 review 时存在
    "review_content": "..."     // 评审意见
  },
  "usage": {                     // MessageBubble 实际渲染
    "tokens_used": 1234,
    "time_elapsed": 8.5,
    "model": "gpt-4o"
  }
  // ❌ 不下发：_stages / _thinkingSteps / _llmChunks / _reasoningChunks
  //            _draftContent / agent_events（运行时缓冲 + 全量 dump）
}
```

### DB 持久化 schema（不变）

DB 入库时由前端 store 拼接运行时缓冲（历史回放需要），**与历史消息兼容**：

```jsonc
{
  ...上面业务轨字段,
  "_stages": [...],
  "_thinkingSteps": [...],
  "_llmChunks": {...},
  "_reasoningChunks": {...},
  "_draftContent": "..."
}
```

### 迁移路径

| 阶段 | 时间 | 工作 |
|------|------|------|
| Phase 1（本周） | 2026-07-01 | ✅ 后端实现 + 单元测试（24 用例） |
| Phase 2（本周） | 2026-07-01 | ✅ 前端切到 v2 + StageTimeline 清理 |
| Phase 3 | 2026-08-XX | 加 `Sunset` HTTP header + 文档提醒 |
| Phase 4 | 2026-09-30 | 删除旧 `/agent/stream` 路由 |

完整设计文档：[docs/streaming-frontend-contract.md](./docs/streaming-frontend-contract.md)
设计 spec：[../docs/superpowers/specs/2026-07-01-streaming-frontend-contract.md](../docs/superpowers/specs/2026-07-01-streaming-frontend-contract.md)

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

---

