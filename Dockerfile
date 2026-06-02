# syntax=docker/dockerfile:1
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建数据库目录
RUN mkdir -p /app/data

# 环境变量
ENV PYTHONUNBUFFERED=1

# 暴露端口（Railway 动态分配 $PORT）
EXPOSE 8000

# 启动命令 — 使用 $PORT 环境变量（Railway 自动注入）
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}
