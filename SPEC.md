# AutoPublish — 项目规格说明

## 项目概述

AI 驱动的内容生产与多平台发布工具。支持从选题、写作、SEO 优化、审核到多平台适配的全流程自动化。

## 技术栈

- **后端**: Python 3.11+, FastAPI, LangChain, LangGraph, SQLAlchemy, ChromaDB
- **前端**: React 19, TypeScript, Vite
- **LLM**: 支持 OpenAI / 智谱 / DeepSeek 等多模型路由

## 项目结构

```
autoGitub/
├── SPEC.md                 # 本文件
├── README.md               # 项目说明
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板
├── .env                    # 实际环境变量
├── demo.py                 # 快速演示脚本
│
├── src/                    # 后端源码
│   ├── main.py             # FastAPI 应用入口
│   ├── config.py           # 配置管理
│   ├── database.py         # 数据库连接
│   ├── api/                # REST API 路由
│   ├── core/               # 核心组件 (安全、LLM 网关、模型路由)
│   ├── models/             # ORM 数据模型
│   ├── schemas/            # Pydantic 请求/响应模型
│   ├── services/           # 业务逻辑服务
│   ├── adapters/           # 多平台适配器
│   ├── agent/              # Agent 编排系统 (LangGraph)
│   ├── tools/              # Agent 工具集
│   ├── prompts/            # Prompt 模板
│   ├── memory/             # 对话记忆管理
│   └── utils/              # 工具函数
│
├── tests/                  # 测试套件
├── frontend/               # 前端源码
├── logs/                   # 日志目录
└── docs/                   # 文档目录
```

## 启动方式

```bash
# 后端
pip install -r requirements.txt
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# 前端
cd frontend && npm install && npm run dev
```

## API 端点

- `GET  /health` — 健康检查
- `POST /api/v1/auth/register` — 用户注册
- `POST /api/v1/auth/login` — 用户登录
- `GET  /api/v1/contents` — 内容列表
- `POST /api/v1/contents/generate` — 生成内容
- `POST /api/v1/agent/run` — 运行 Agent
- `POST /api/v1/agent/async/submit` — 异步 Agent 任务

## 环境变量

| 变量 | 说明 |
|------|------|
| `DATABASE_URL` | 数据库连接字符串 |
| `OPENAI_API_KEY` | OpenAI API 密钥 |
| `ZHIPU_API_KEY` | 智谱 API 密钥 |
| `SECRET_KEY` | JWT 签名密钥 |
| `CHROMA_PERSIST_DIR` | ChromaDB 持久化目录 |
