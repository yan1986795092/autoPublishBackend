# AutoPublish — 产品与系统设计规格

> **文档状态**: Draft v1  
> **版本**: 3.0.0  
> **最后更新**: 2026-06-26  
> **负责人**: TBD  
> **对标格式**: Google Design Doc / Stripe API Spec / Microsoft Agent Schema

---

## 目录

1. [Context & Scope](#1-context--scope)
2. [Goals & Non-Goals](#2-goals--non-goals)
3. [System Overview](#3-system-overview)
4. [User Scenarios](#4-user-scenarios)
5. [Agent System Design](#5-agent-system-design)
6. [Interaction Protocol](#6-interaction-protocol)
7. [Platform Adaptation Model](#7-platform-adaptation-model)
8. [Memory & Context](#8-memory--context)
9. [LLM Routing Strategy](#9-llm-routing-strategy)
10. [Cross-Cutting Concerns](#10-cross-cutting-concerns)
11. [Alternatives Considered](#11-alternatives-considered)
12. [Appendix: API Reference](#12-appendix-api-reference)

---

## 1. Context & Scope

### 1.1 产品定位

AutoPublish 是一个 **AI 驱动的多 Agent 内容生产系统**。它不是传统的内容管理后台，而是一个"内容工厂流水线"——从选题研究、热点追踪、内容创作、质量审核到多平台发布，全流程由 Agent 协作完成。

核心命题：**如何让一个人像一支内容团队一样高效运作。**

### 1.2 解决的问题

| 角色 | 痛点 | AutoPublish 方案 |
|------|------|-----------------|
| 自媒体创作者 | 每天写 3+ 平台，重复劳动 | 一次生成，多平台适配 |
| 内容运营 | 追热点+写稿+审核，精力分散 | Agent 自动搜索→分析→写作→审核 |
| 企业市场部 | 品牌内容一致性难保证 | PlatformProfile 统一品牌规则 |
| 独立开发者 | 技术内容生产耗时 | CodeAgent + DiagramAgent 辅助 |

### 1.3 系统边界

**在范围内**:
- 多 Agent 协作的内容生成与优化
- 多平台内容适配（小红书、公众号、知乎等 15 种形态）
- 实时流式生成体验（SSE）
- 异步任务管理与死信队列
- 联网研究（WebSearch）与热点分析
- 内容版本管理与历史
- 技能模板（Skill）的复用与导入导出

**不在范围内**:

- 实际发布到第三方平台的 API 集成（由前端/外部服务处理）
- 多用户协作工作流
- A/B 测试或内容效果分析
- 媒体资产管理（图片/视频上传）
- 社交账号授权管理
- 批量定时发布调度

---

## 2. Goals & Non-Goals

### 2.1 Goals

| # | 目标 | 衡量标准 |
|---|------|---------|
| G1 | 用户输入一个主题，系统自动完成研究→写作→审核→优化全流程 | 端到端成功率 > 90% |
| G2 | 生成内容质量达到"可用"水平（> 70% 用户不修改直接发布） | ReviewAgent 通过率 > 70% |
| G3 | 用户可在 3s 内看到 Agent 开始思考的反馈 | SSE 首帧延迟 < 3s |
| G4 | 一次生成可适配 6 个发布平台 | PlatformProfile 覆盖 15 种内容形态 |
| G5 | 异步任务不丢失，失败可重试 | DLQ + 重试机制 |
| G6 | 新增一个 Agent 只需注册 + 实现 Handler，不改编排逻辑 | AgentCard 注册模式 |

### 2.2 Non-Goals

这些是有意**不做**的事情（不是遗漏，是设计选择）：

- **不追求 LLM 生成质量的无限提升**：Review-Fix 循环最多 2 轮，超过则认为当前模型能力边界已到，不无限浪费 Token
- **不做 Agent 间自由对话**：所有 Agent 通过共享的 `AgentGraphState` 通信，不引入独立的 Agent-to-Agent Message Bus
- **不做实时协作编辑**：多人同时编辑同一篇内容不在当前范围
- **不做内容调度排期**：定时发布由前端/后端协作完成，非 Agent 编排系统的职责
- **不做多模态生成**：不支持图片/视频生成，但可通过 Pexels MCP 检索图片
- **不做 Agent 记忆的长期持久化**：记忆系统以会话为中心，跨会话记忆有限（语义记忆 + 情景记忆），不追求"完美终身记忆"

---

## 3. System Overview

### 3.1 一句话架构

```
Frontend (React) ↔ AutoPublish API (FastAPI, :9000)
                      ↕ LangGraph Agent Orchestration
                      ↕ ChatOpenAI Protocol
                   LLM Gateway (:9010)
                      ↕ httpx
                   OpenAI / 智谱 / 通义千问 / DeepSeek
```

### 3.2 系统组件及职责

```
┌─────────────────────────────────────────────────────────────────────┐
│                     用户交互层                                       │
│                                                                     │
│  Frontend (React 18, Vite)                                          │
│  ├── Agent 工作室: 聊天界面 + 实时 Agent Timeline                    │
│  ├── 内容管理: 列表/编辑/版本/日历                                    │
│  ├── 设置: Skills / Platforms / API Keys                            │
│  └── 运维中心: 成本/发布记录/系统监控                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ REST + SSE + WebSocket
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     编排与执行层                                      │
│                                                                      │
│  API 路由 (FastAPI, :9000)                                           │
│  ├── 内容服务: CRUD / 生成 / 版本 / 日历                              │
│  ├── Agent 服务: 同步/流式/异步/WebSocket 四种模式                    │
│  ├── 对话服务: Chat CRUD / ConversationEngine                        │
│  ├── 编辑器服务: 14 个 AI 编辑功能                                   │
│  └── 辅助服务: Auth / Skills / Publish / Platforms / Admin           │
│                                                                      │
│  Agent 编排系统 (LangGraph StateGraph)                               │
│  ├── Orchestrator: 5 节点 DAG (理解→规划→调度→执行→合并)              │
│  ├── Handler 池: 8 个注册 Handler + 扩展                            │
│  ├── AgentCard 注册表: 元数据驱动                                    │
│  └── Task Queue: 异步提交/执行/重试/DLQ                              │
│                                                                      │
│  记忆系统                                                            │
│  ├── WorkingMemory: 会话内上下文                                     │
│  ├── SemanticMemory: 向量语义检索                                    │
│  └── EpisodicMemory: 历史记录                                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ ChatOpenAI (base_url=:9010)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     LLM 网关层 (:9010)                                │
│                                                                      │
│  LLM Gateway                                                         │
│  ├── Plugin Chain: RateLimiter → Cache → CircuitBreaker → Obs        │
│  ├── AutoRouter: model=auto 按成本路由                               │
│  └── Adapters: OpenAI / 智谱 / 通义千问 / DeepSeek                   │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.3 数据流

**典型写一篇小红书笔记**:

```
User: "写一篇露营装备推荐的小红书"
   │
   ▼
1. ManagerAgent 理解意图 → {平台: 小红书, 类型: social_post, 语气: 口语化}
   │
   ▼
2. ManagerAgent 拆解任务 → [WebSearch → WritingAgent → ReviewAgent → FixAgent]
   │
   ▼
3. WebSearch 搜索 "露营装备推荐 2026" → 返回搜索结果
   │
   ▼
4. WritingAgent 创作 → 800 字笔记 (emoji + 口语化 + 种草结构)
   │
   ▼
5. ReviewAgent 审核 → 检查违禁词/可读性/平台规则 → passed: false, feedback: "需增加价格信息"
   │
   ▼
6. FixAgent 优化 → 补充价格段 + 优化开头 hook
   │
   ▼
7. ReviewAgent 复审 → passed: true
   │
   ▼
8. 合并 → 写入 Content 表 → 返回最终结果
   │
   ▼
User 收到 SSE 结果 → 可在编辑器中进一步修改 → 一键发布
```

---

## 4. User Scenarios

### 4.1 核心场景

**S1: 一句话内容生成**

```
用户输入: "写一篇 AI 编程工具推荐的文章，发在知乎"
系统行为:
  1. 理解意图 → 知乎长文, 专业偏技术
  2. WebSearch 搜索当前主流 AI 编程工具
  3. WritingAgent 生成 2000+ 字深度对比
  4. ReviewAgent 审核 → FixAgent 优化
  5. 返回结果, 知乎格式适配
用户感知: 2-3 分钟后收到可直接发布的文章
```

**S2: 热点追文**

```
用户输入: "帮我追一下今天的 AI 热点，写个公众号"
系统行为:
  1. WebSearch 搜索今日 AI 热点
  2. HotTopicAgent 分析趋势 + 切入角度建议
  3. WritingAgent 基于分析结果创作
  4. 适配公众号格式 (标题党+深度+分段)
用户感知: 30s 内看到搜索过程, 3-5 分钟完成全流程
```

**S3: 内容再利用**

```
用户操作: 选中一篇已发布的公众号长文 → "改写成小红书笔记"
系统行为:
  1. 读取原文内容
  2. 分析核心信息 + 提取爆点
  3. 小红书风格重写 (短句 + emoji + 种草结构)
用户感知: 原封不动的内容被"翻译"成平台风格
```

**S4: 异步批量生产**

```
用户操作: 提交 5 个选题 → 关闭页面
系统行为:
  1. 5 个任务依次排队执行
  2. AgentWorker 逐个消费
  3. 完成后状态更新, 用户回来看到结果
用户感知: 不等待, 回来后看到 5 篇草稿
```

### 4.2 交互模式

| 模式 | 输入方式 | 响应方式 | 适用场景 |
|------|---------|---------|---------|
| **同步** | HTTP POST | 等待返回完整结果 | 快速生成、预览计划 |
| **SSE 流式** | HTTP POST | SSE 事件流 | 实时看到 Agent 思考过程（推荐） |
| **异步提交** | HTTP POST | 立即返回 task_id | 长时间任务、批量生产 |
| **WebSocket** | WS 连接 | 双向实时推送 | 需要前端主动查询的交互 |

---

## 5. Agent System Design

### 5.1 Agent 是什么

在本系统中，Agent 是一个**具有明确输入/输出契约的自动化内容处理单元**。每个 Agent 由三部分组成：

```
Agent = AgentCard (元数据) + Handler (执行逻辑) + LLM (大脑)
```

Agent 之间**不直接对话**，通过共享的 `AgentGraphState` 交换数据。

### 5.2 Agent 团队

| Agent | 角色 | 技能 | 输入 | 输出 |
|-------|------|------|------|------|
| **ManagerAgent** | 主编 | 意图理解 + 任务规划 | 用户请求 | 执行计划 + 任务链 |
| **WritingAgent** | 主笔 | 多风格内容创作 | topic + 平台 + 语气 | 正文 (Markdown) |
| **ReviewAgent** | 审核编辑 | 质量检查 + 违禁词 | content + 平台规则 | passed + feedback |
| **FixAgent** | 责任编辑 | 按反馈优化 | content + feedback | 优化版正文 |
| **WebSearch** | 研究员 | 联网搜索 | query | 搜索结果 |
| **URLCrawler** | 数据采集 | 页面抓取 | url | 页面正文 |
| **HotTopicAgent** | 分析师 | 热点趋势分析 | 搜索结果 | 分析洞察 |
| **DiagramAgent** | 美术 | Mermaid 图表 | topic | 图表代码 |
| **CodeAgent** | 程序员 | 代码/技术文档 | task | 代码片段 |

### 5.3 编排流程 (LangGraph 5 节点)

```
用户请求
    │
    ▼
┌─────────────────────────────────────────┐
│ 1. understand_intent                     │
│    主编(ManagerAgent) 理解用户想做什么    │
│    输出: {intent, platform, slots}       │
│    兜底: 关键词规则匹配                  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 2. plan_tasks                            │
│    主编决定派哪些 Agent, 按什么顺序       │
│    输出: ExecutionPlan {tasks, deps}     │
│    兜底: 分类+关键词 → Agent 映射        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ 3. schedule_and_execute ──── 条件判断 ──►│
│    检查谁准备好了 (依赖已满足)            │
│    死锁检测                              │
└──────────┬──────────────────────────────┘
           │ 有就绪任务
           ▼
┌─────────────────────────────────────────┐
│ 4. execute_task_batch                    │
│    并行执行 (最多 3 个并发)              │
│    ← 审核未过? → 自动追加 Review-Fix 循环│
│    ← 都做完了? → 进入合并               │
└──────────┬──────────────────────────────┘
           │ 全部完成
           ▼
┌─────────────────────────────────────────┐
│ 5. merge_and_finish                      │
│    Fix > ReviewFix > Writing 择优合并     │
│    自动提取标题 + 质量过滤               │
│    异步写入 DB (Content + AgentTask)     │
└─────────────────────────────────────────┘
    │
    ▼
返回结果给用户
```

### 5.4 Review-Fix 循环

这是系统最核心的**质量闭环**：

```
WritingAgent → ReviewAgent ── passed ──→ 直接进入合并
                              │
                           not passed
                              │
                              ▼
                          FixAgent (修改)
                              │
                              ▼
                          ReviewAgent (复审, 第 2 轮)
                              │
                         passed / 已达最多 2 轮 → 进入合并
```

**设计理由**:
- 一次 Review-Fix 多数情况下够用
- 两轮后质量提升边际递减
- 限制轮数防止 Token 无穷消耗
- 复审通过后 FixAgent 被自动跳过（短路优化）

### 5.5 AgentCard 注册模式

新增 Agent 只需做三件事：

```python
# 1. 实现 Handler 函数
async def handle_my_agent(task_input: dict, llm, mcp_tools) -> dict:
    ...

# 2. 注册到 HANDLER_MAP
HANDLER_MAP["MyAgent"] = handle_my_agent

# 3. 注册 AgentCard 元数据
register(AgentCard(
    name="MyAgent",
    description="...",
    category="writing",     # 决定依赖链位置
    model="glm-4-flash",    # 默认模型
    stage_label="...",      # 前端展示
    stage_icon="...",
    stage_color="...",
    input_schema={...},     # LLM 规划时知道你的输入格式
))
```

编排器、前端展示、依赖链全部自动感知，**零侵入**。

---

## 6. Interaction Protocol

### 6.1 四种执行模式

```
┌─────────────┬────────────┬──────────────┬─────────────────────┐
│   模式      │ 端点        │ 响应方式      │ 典型使用场景          │
├─────────────┼────────────┼──────────────┼─────────────────────┤
│ 同步执行    │ /agent/run │ JSON 完整结果 │ 快速生成, 预览计划    │
│ SSE 流式    │ /agent/stream │ text/event-stream │ 实时查看 Agent 过程  │
│ 异步提交    │ /agent/async/submit │ task_id │ 长时间任务, 离开页面 │
│ WebSocket   │ /agent/ws/{id} │ 双向帧     │ 需要实时查询的交互    │
└─────────────┴────────────┴──────────────┴─────────────────────┘
```

### 6.2 SSE 事件规范

**连接方式**: `POST /api/v1/agent/stream` — 请求体包含 prompt + context

**事件流** (按发送顺序):

```json
// stage: Agent 阶段开始/结束 (前端 Timeline 用)
event: stage
data: {"agent":"ManagerAgent","label":"思考分析中","icon":"🧠","status":"started"}

// reasoning: Agent 的推理过程 (前端思考气泡用)
event: reasoning
data: {"agent":"ManagerAgent","content":"📋 意图分析：CREATE - generate content\n📍 目标平台：小红书..."}

// llm_chunk: LLM 流式输出 (前端实时渲染用)
event: llm_chunk
data: {"agent":"WritingAgent","content":"露营新手必看", "channel":"body"}

// thinking_step: 每个 Agent 步骤的起止
event: thinking_step
data: {"agent":"WritingAgent","action":"开始执行","status":"running"}

// progress: 整体进度百分比
event: progress
data: {"progress":60,"message":"审核优化中"}

// result: 最终结果
event: result
data: {"output":{"full_text":"...","sections":[...]},"status":"completed"}

// error: 失败
event: error
data: {"error":"Agent 执行超时","code":"TIMEOUT"}
```

**事件类型枚举**:

| 事件 | 前端用途 | 发送者 | 是否必达 |
|------|---------|--------|---------|
| `stage` | Timeline 显示 Agent 阶段 | Orchestrator | 是 |
| `reasoning` | 思考气泡展示 | ManagerAgent | 建议 |
| `llm_chunk` | 实时文字渲染 | Handler | 是(流式) |
| `thinking_step` | Agent 步骤详情 | Handler | 建议 |
| `progress` | 进度条 | Orchestrator | 建议 |
| `result` | 最终结果展示 | Orchestrator | 是 |
| `error` | 错误提示 | Orchestrator | 是 |

### 6.3 异步任务生命周期

```
submit → PENDING → RUNNING → COMPLETED
                         │
                         ├→ FAILED → DLQ → (可选重试) → PENDING
                         │
                         └→ CANCELLED
```

- `GET /agent/async/status/{id}`: 任意时刻查询
- `DELETE /agent/async/{id}`: 取消（仅 PENDING/RUNNING 状态）
- `GET /agent/dlq`: 查看死信队列
- `POST /agent/dlq/{id}/retry`: 从 DLQ 重新提交

---

## 7. Platform Adaptation Model

### 7.1 问题

同一个内容在不同平台需要完全不同的写作风格：

> "露营装备推荐"
> - **小红书**: emoji 开头 + 口语化 + 300-800 字 + 种草结构
> - **知乎**: 专业对比 + 数据支撑 + 2000-4000 字 + 深度分析
> - **公众号**: 标题党 + 故事化开头 + 分段 + 1500-3000 字
> - **抖音/B站**: 口播脚本 + hook 开头 + 分镜标注

传统方案是每家平台写一套 Prompt 硬编码。AutoPublish 的方案是 **PlatformProfile**——每个平台/内容类型的完整写作画像。

### 7.2 PlatformProfile 结构

```python
@dataclass
class PlatformProfile:
    code: str                    # "xiaohongshu"
    name: str                    # "小红书"
    category: str                # "social_media"
    word_count_range: (int, int) # (300, 1000)
    tone_tags: [str]             # ["casual", "friendly", "emotional"]
    target_audience: str         # "18-35岁女性"
    system_prompt: str           # 平台 System Prompt (写入 LLM 角色)
    writing_rules: [str]         # 写作规则列表
    example_structures: [dict]   # 示例结构
    seo_rules: dict              # SEO 改写规则 (可选)
    quality_checks: [dict]       # 质量检查规则
```

### 7.3 当前覆盖 (15 种形态)

| 分类 | 形态 | 平台代码 |
|------|------|---------|
| **社交平台** (6) | 小红书、微信公众号、微博、抖音、B站、知乎 | `xiaohongshu`, `wechat`, `weibo`, `douyin`, `bilibili`, `zhihu` |
| **内容创作** (5) | 产品文案、品牌故事、广告文案、小说故事、剧本脚本 | `copywriting`, `brand-story`, `ad-copy`, `fiction`, `script` |
| **专业写作** (4) | 技术文档、教程指南、SEO文章、邮件营销 | `tech-doc`, `tutorial`, `seo`, `email` |

### 7.4 平台选择流程

```
用户请求
    │
    ▼
1. 前端 PlatformSelector → 传递 platform 参数
    │
    ├── 有指定 → 使用对应 PlatformProfile
    │
    └── 未指定 → understand_intent LLM 检测平台
        │
        ├── 检测到 → 使用对应 PlatformProfile
        │
        └── 未检测到 → 默认 blog 类型

2. Profile 被注入:
   ├── task_input._platform_system_prompt → LLM System Prompt
   ├── task_input._platform_writing_rules → 写入规则
   └── task_input._platform_quality_check → ReviewAgent 审核标准
```

---

## 8. Memory & Context

### 8.1 设计目标

- **会话内**: 当前对话上下文的完整传递
- **跨会话**: 记住用户偏好、历史内容风格
- **不追求完美**: 允许遗忘，有限 Token 预算下做最优检索

### 8.2 三层架构

```
WorkingMemory (进程内存)
  ├── 当前会话 ID / 用户 ID
  ├── Token 预算追踪
  ├── 前端传入的结构化参数
  └── 生命周期: 一次 Agent 执行

SemanticMemory (向量存储)
  ├── 用户写作偏好 (平台/语气/风格)
  ├── 高质量历史内容片段
  ├── 语义检索 → 注入 Agent Context
  └── 生命周期: 持久化, 跨会话

EpisodicMemory (SQLite)
  ├── Agent 输入/输出摘要
  ├── 最近执行历史
  ├── 后台 Consolidator 定期压缩
  └── 生命周期: 持久化, 有上限
```

### 8.3 关键流程

```
Agent 执行前:
  MemoryManager.build_agent_context()
    → SemanticMemory.search(当前任务)
    → 找到相关的历史偏好/内容
    → 注入 task_input._memory_context

Agent 执行后 (产出型 Agent 如 WritingAgent):
  MemoryManager.remember()
    → 质量过滤 (低质量内容不沉淀)
    → SemanticMemory.store(向量化)
    → EpisodicMemory.store(摘要)
```

### 8.4 上下文注入策略

```python
# Task 输入中包含的上下文来源
task_input = {
    "_context": {                               # 依赖的上游任务结果
        "task_id_1": {"content":"...", "agent":"WritingAgent"},
    },
    "_memory_context": "...",                    # 记忆系统检索的相关上下文
    "_agent_context": {                          # 前端传入的结构化参数
        "topic":"露营", "tone":"casual", ...
    },
    "platform": "xiaohongshu",                   # 平台选择
    "_platform_system_prompt": "...",             # 平台 System Prompt
    "_platform_writing_rules": [...],             # 平台写作规则
}
```

---

## 9. LLM Routing Strategy

### 9.1 问题

- 不同 Agent 需要不同能力的模型（Manager 要高推理，搜索要低成本）
- 不同 Provider 价格差异大
- 某个 Provider 可能故障

### 9.2 方案: Per-Role Model Router

```
Config
├── ManagerAgent → deepseek-chat (高推理)
├── WritingAgent → auto (按成本路由)
├── ReviewAgent  → auto (分析型)
├── WebSearch    → auto (低成本)
└── CodeAgent    → deepseek-chat (代码)

回退链:
  角色专用 Key → Provider 默认 Key → 遍历全部 Provider
```

### 9.3 两种路由模式

```python
# 模式 1: 自动路由 (按 X-Route-Hint 选最便宜的 Provider)
get_chat_model("WritingAgent")
# → ChatOpenAI(model="auto", headers={"X-Route-Hint": "creative"})

# 模式 2: 指定模型 (高推理 Agent)
get_chat_model_for_model("deepseek-chat", "ManagerAgent")
# → ChatOpenAI(model="deepseek-chat")
```

### 9.4 Provider 支持

| Provider | 环境变量 | 默认模型 | 自动路由 |
|----------|---------|---------|---------|
| OpenAI | `OPENAI_API_KEY` | gpt-3.5-turbo | ✅ |
| 智谱 | `ZHIPU_API_KEY` | glm-4-flash | ✅ |
| 通义千问 | `QWEN_API_KEY` | qwen-plus | ✅ |
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat | ✅ |

### 9.5 调用架构

```
Agent Handler
    │
    ├── ChatOpenAI(base_url="http://localhost:9010/v1")
    │                               │
    │                        LLM Gateway (:9010)
    │                         ├── RateLimiter
    │                         ├── Cache (Redis)
    │                         ├── CircuitBreaker
    │                         └── AutoRouter → Provider Adapter
    │                                               │
    │                                   OpenAI / 智谱 / 通义 / DeepSeek
    │
    └── MCP Tool (不经过 Gateway)
        WebSearch / Jina / Brave / ...
```

---

## 10. Cross-Cutting Concerns

### 10.1 安全性

| 层次 | 措施 |
|------|------|
| 认证 | JWT Bearer Token (Access 24h + Refresh 7d) |
| 密码 | bcrypt (passlib) |
| 租户隔离 | ContextVar + ORM 自动注入 user_id 过滤 |
| 输入验证 | Pydantic schema 校验 |
| URL 安全 | 仅允许 http/https 协议的公开 URL |
| 异常安全 | 生产环境不暴露内部堆栈，统一错误格式 |
| CORS | 白名单控制（前端 dev server + Gateway） |

**租户隔离机制**:

```python
# 请求进入 → get_current_user() → set_tenant(user.id)
# SQLAlchemy Query → ORMTenantGuard 自动添加 user_id 过滤
# 请求结束 → clear_tenant()
# 后台任务 → run_as_system() 临时禁用隔离
```

### 10.2 可观测性

**三层数据结构**:

```
Traces (traces-YYYY.MM.DD)
  ├── AgentTracer 轻量 span
  ├── 记录每次 Agent 执行的耗时和状态
  └── JSON 直接写 ES

Agent Events (agent-events-YYYY.MM.DD)
  ├── AgentEventEmitter 发送
  ├── 记录推理过程、工具调用、Token 成本
  └── 用于调试 Agent 行为

Logs (logs-YYYY.MM.DD)
  ├── structlog 结构化日志
  ├── 带 trace_id 跨组件关联
  └── 问题排查和审计
```

**写入策略**: ES 不可用时自动回退到 JSONL 文件，不阻塞主流程。

**错误处理**: 11 类异常处理器，全部返回统一格式：

```json
{"error": {"code": "ERROR_CODE", "message": "用户可读的描述", "details": {}}}
```

### 10.3 性能

| 指标 | 目标 | 约束 |
|------|------|------|
| SSE 首帧 | < 3s | 取决于 LLM 首 Token 延迟 |
| 单 Agent 执行 | < 30s | 取决于模型和输入长度 |
| 完整流程 (4+ Agent) | < 180s | Orchestrator 超时 |
| 并发 Agent 数 | 3 (可配置) | 防止 Token 爆炸和 API 限流 |

### 10.4 可扩展性

**新增 Provider**: 只需在 config.py 添加配置项 + LLM Gateway 添加 Adapter

**新增 Agent**: 实现 Handler → 注册 AgentCard → 自动生效（见 5.5）

**新增平台**: 在 `platform_profile.py` 添加 PlatformProfile

**水平扩展**: 多实例部署 + 替换 asyncio.Queue 为 Redis Queue

---

## 11. Alternatives Considered

### 11.1 Agent 通信方式

| 方案 | 考虑过 | 未采用原因 |
|------|--------|-----------|
| **共享 State** (当前方案) | ✅ 采用 | 简单可靠, LangGraph 原生支持 |
| Agent-to-Agent Message Bus | ❌ 未采用 | 增加复杂度, 当前场景不需要 Agent 间自由对话 |
| 集中式 Database 通信 | ❌ 未采用 | 增加延迟, 不适合实时流式场景 |

### 11.2 编排引擎

| 方案 | 考虑过 | 未采用原因 |
|------|--------|-----------|
| **LangGraph StateGraph** (当前方案) | ✅ 采用 | 原生 DAG + 条件边, Python 生态 |
| 自定义 DAG 调度器 | ❌ 未采用 | 重复造轮子, 缺少条件边能力 |
| Prefect / Temporal | ❌ 未采用 | 太重, 适合数据管道而非 Agent 编排 |
| 单 Agent 顺序调用 | ❌ 未采用 | 无法并行, 无循环, 不可扩展 |

### 11.3 LLM 调用路径

| 方案 | 考虑过 | 未采用原因 |
|------|--------|-----------|
| **统一 Gateway** (当前方案) | ✅ 采用 | 集中治理限流/熔断/缓存 |
| 直连 Provider | ❌ 未采用 | 每个 Provider 独立管理, 重复代码 |
| 仅用 LiteLLM | ❌ 未采用 | Gateway 提供更细粒度的路由控制 |

### 11.4 流式推送方案

| 方案 | 考虑过 | 未采用原因 |
|------|--------|-----------|
| **SSE + EventBus** (当前方案) | ✅ 采用 | 单向推送, 适合 Agent→前端场景 |
| WebSocket Only | ❌ 未采用 | 双向能力过剩, 需要维护连接状态 |
| 轮询 | ❌ 未采用 | 延迟高, 浪费资源 |
| Server-Sent Events + WS | ✅ 采用 | SSE 做主要推送, WS 做补充双向 |

### 11.5 Tenant 隔离

| 方案 | 考虑过 | 未采用原因 |
|------|--------|-----------|
| **ContextVar + ORM Hook** (当前方案) | ✅ 采用 | 零侵入, async 安全 |
| threading.local | ❌ 未采用 | async 下不跨 await 传播 |
| 每个请求创建独立 Session | ❌ 未采用 | Session 管理成本高 |
| 行级 SQL 过滤 | ❌ 未采用 | 需要在所有查询手工加 WHERE |

### 11.6 记忆系统方案

| 方案 | 考虑过 | 未采用原因 |
|------|--------|-----------|
| **三层记忆** (当前方案) | ✅ 采用 | 兼顾会话内 + 跨会话需求 |
| 纯向量记忆 | ❌ 未采用 | 缺少工作记忆, 不适合会话内上下文 |
| 纯工作记忆 | ❌ 未采用 | 每次冷启动, 浪费 Token |
| LangChain Memory | ❌ 未采用 | 不够灵活, 难以定制存储策略 |

### 11.7 Review-Fix 轮次上限

| 方案 | 考虑过 | 未采用原因 |
|------|--------|-----------|
| **最多 2 轮** (当前方案) | ✅ 采用 | 第 3 轮后质量提升 < 5%, 成本线性增长 |
| 无限制直到通过 | ❌ 未采用 | 极端情况下 Token 无限消耗, 平均需要 1.7 轮 |
| 固定 1 轮 | ❌ 未采用 | 约 30% 场景一轮不足 |
| 动态判断 | ❌ 未采用 | 引入额外 LLM 调用判断是否继续, 性价比低 |

---

## 12. Appendix: API Reference

### 12.1 Agent 端点

| 方法 | 路径 | 说明 | 请求体 | 响应 |
|------|------|------|--------|------|
| POST | `/api/v1/agent/run` | 同步执行 | `{prompt, context?, skill_name?}` | `AgentResult` |
| POST | `/api/v1/agent/plan` | 预览计划 | `{prompt, context?}` | `ExecutionPlan` |
| POST | `/api/v1/agent/stream` | SSE 流式 | `{prompt, context?, skill_name?}` | `text/event-stream` |
| POST | `/api/v1/agent/async/submit` | 异步提交 | `{prompt, context?, skill_name?}` | `{task_id}` |
| GET | `/api/v1/agent/async/status/{id}` | 查询状态 | — | `AsyncTask` |
| GET | `/api/v1/agent/async/list` | 任务列表 | `?page&size` | `{tasks, total}` |
| DELETE | `/api/v1/agent/async/{id}` | 取消任务 | — | `{status}` |
| POST | `/api/v1/agent/repurpose` | 内容再利用 | `{content, target_platforms}` | `AgentResult` |
| POST | `/api/v1/agent/crawl` | URL 抓取 | `{url, analyze?}` | `CrawlResult` |
| GET | `/api/v1/agent/dlq` | 死信队列 | — | `[AsyncTask]` |
| POST | `/api/v1/agent/dlq/{id}/retry` | DLQ 重试 | — | `{new_task_id}` |
| POST | `/api/v1/agent/vector-search` | 向量搜索 | `{query, limit?}` | `[SearchResult]` |
| GET | `/api/v1/agent/cache/stats` | 缓存统计 | — | `CacheStats` |
| GET | `/api/v1/agent/metrics` | 监控指标 | — | `AgentMetrics` |
| WS | `/api/v1/agent/ws/{id}` | WebSocket | JWT Query Param | 事件帧 |

### 12.2 内容端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/contents/generate` | AI 生成 (同步) |
| POST | `/api/v1/contents/generate/stream` | AI 生成 (SSE 流式) |
| POST | `/api/v1/contents/generate/batch` | 批量生成 |
| GET | `/api/v1/contents` | 列表 (分页+统计) |
| GET | `/api/v1/contents/calendar` | 内容日历 |
| GET/PUT/DELETE | `/api/v1/contents/{id}` | 内容 CRUD |
| PUT | `/api/v1/contents/{id}/autosave` | 自动保存 |
| POST | `/api/v1/contents/{id}/check-banned-words` | 敏感词检测 |
| GET | `/api/v1/contents/{id}/versions` | 版本历史 |
| GET | `/api/v1/contents/{id}/versions/diff` | 版本对比 |

### 12.3 AI 编辑器端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/contents/ai-edit/rewrite` | 重写优化 |
| POST | `/api/v1/contents/ai-edit/expand` | 内容扩展 |
| POST | `/api/v1/contents/ai-edit/improve-tone` | 语气改善 |
| POST | `/api/v1/contents/ai-edit/summarize` | 内容摘要 |
| POST | `/api/v1/contents/ai-edit/translate` | 翻译 |
| POST | `/api/v1/contents/ai-edit/check-banned-words` | 敏感词检测 |
| POST | `/api/v1/contents/ai-edit/check-banned-words-ai` | AI 敏感词检测 |
| POST | `/api/v1/contents/ai-edit/suggest-tags` | 标签推荐 |
| POST | `/api/v1/contents/ai-edit/generate-headlines` | 标题生成 |
| POST | `/api/v1/contents/ai-edit/analyze-seo` | SEO 分析 |
| POST | `/api/v1/contents/ai-edit/adapt-content` | 平台适配 |
| POST | `/api/v1/contents/ai-edit/expand-keywords` | 关键词扩展 |
| POST | `/api/v1/contents/ai-edit/analyze-viral` | 爆款分析 |
| POST | `/api/v1/contents/ai-edit/parse-url` | URL 解析 |

### 12.4 数据模型

**AgentResult** (编排执行结果):

```json
{
  "task_id": "uuid",
  "status": "completed | partial | failed | timeout",
  "agent_type": "ManagerAgent",
  "output": {
    "goal": "写一篇露营文章",
    "sections": [{"agent": "WritingAgent", "output": {"content": "..."}}],
    "full_text": "最终合并的完整正文",
    "title": "自动提取的标题",
    "platform": "xiaohongshu"
  },
  "error": null,
  "usage": {
    "tokens_used": 4521,
    "time_elapsed": 45.2,
    "model": "deepseek-chat"
  }
}
```

**ExecutionPlan** (预览计划):

```json
{
  "goal": "写一篇露营文章",
  "tasks": [
    {"id": "uuid", "agent_type": "WebSearch", "input": {"query": "..."}, "dependencies": []},
    {"id": "uuid", "agent_type": "WritingAgent", "input": {"topic": "..."}, "dependencies": ["task_1"]},
    {"id": "uuid", "agent_type": "ReviewAgent", "input": {}, "dependencies": ["task_2"]}
  ],
  "reasoning": "先搜索资料 → 写作 → 审核",
  "estimated_steps": 3,
  "plan_method": "llm | template"
}
```

**AsyncTask** (异步任务状态):

```json
{
  "id": "uuid",
  "state": "pending | running | completed | failed | cancelled",
  "progress": 60,
  "plan": {"goal": "...", "reasoning": "..."},
  "result": null,
  "error": null,
  "created_at": "2026-06-26T10:00:00Z"
}
```

## 
