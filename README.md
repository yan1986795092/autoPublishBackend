# AutoPublish Backend

AI 内容生产与发布工具后端 API 服务，基于 FastAPI 构建。

## 技术栈

- **FastAPI** - Web 框架
- **SQLAlchemy** - ORM
- **SQLite** - 数据库（开发环境）
- **Python-JOSE** - JWT 认证
- **Passlib + bcrypt** - 密码加密
- **Jinja2** - Prompt 模板引擎
- **httpx** - HTTP 客户端
- **Pydantic** - 数据验证

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

## 项目结构

```
backend/
├── app/
│   ├── api/            # API 路由
│   │   ├── auth.py         # 认证（注册/登录/Token刷新）
│   │   ├── contents.py     # 内容 CRUD
│   │   ├── publish.py      # 内容发布
│   │   ├── platforms.py    # 平台管理
│   │   ├── templates.py    # Prompt 模板
│   │   ├── presets.py      # 用户预设
│   │   ├── personas.py     # 人设管理
│   │   └── ai_editor.py    # AI 编辑助手
│   ├── core/           # 核心模块
│   │   ├── llm_gateway.py  # LLM 网关（OpenAI/通义千问/Mock）
│   │   ├── prompt_engine.py # Prompt 模板引擎
│   │   ├── security.py     # JWT + 密码安全
│   │   └── exceptions.py   # 异常定义
│   ├── models/         # SQLAlchemy 数据模型
│   ├── schemas/        # Pydantic 请求/响应 Schema
│   ├── services/       # 业务逻辑层
│   ├── adapters/       # 平台适配器（小红书/公众号/知乎）
│   ├── config.py       # 应用配置
│   ├── database.py     # 数据库连接
│   └── main.py         # 应用入口
├── scripts/            # 工具脚本
├── tests/              # 单元测试
├── .env.example        # 环境变量模板
├── Dockerfile          # Docker 构建
├── docker-compose.yml  # Docker 编排
└── requirements.txt    # Python 依赖
```

## API 概览

所有 API 均挂载在 `/api/v1` 前缀下。

| 模块 | 端点 | 说明 |
|------|------|------|
| 认证 | `POST /auth/register` | 用户注册 |
| 认证 | `POST /auth/login` | 用户登录 |
| 认证 | `POST /auth/refresh` | 刷新 Token |
| 认证 | `GET /auth/me` | 获取当前用户信息 |
| 内容 | `GET/POST /contents` | 内容列表 / 新建 |
| 内容 | `GET/PUT/DELETE /contents/{id}` | 内容详情 / 更新 / 删除 |
| 内容 | `POST /contents/{id}/versions` | 保存版本历史 |
| 发布 | `POST /contents/{id}/publish` | 发布内容到平台 |
| 平台 | `GET /platforms` | 平台列表 |
| 模板 | `GET/POST /templates` | Prompt 模板管理 |
| 预设 | `GET/POST /presets` | 用户预设管理 |
| 人设 | `GET/POST /personas` | 人设管理 |
| AI | `POST /ai/generate` | AI 内容生成 |
| AI | `POST /ai/edit` | AI 编辑优化 |

## 环境变量

主要配置项（详见 `.env.example`）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DATABASE_URL` | 数据库连接 | `sqlite:///./autopublish.db` |
| `SECRET_KEY` | JWT 密钥 | - |
| `OPENAI_API_KEY` | OpenAI API Key（留空=Mock） | - |
| `DASHSCOPE_API_KEY` | 通义千问 API Key | - |

## 运行测试

```bash
pytest tests/ -v
```
