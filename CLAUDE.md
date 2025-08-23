# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 proxy.py 的 LLM 代理服务器，根据 URL 模式将 HTTP 请求路由到不同的 LLM 提供商（OpenAI、Anthropic、Azure OpenAI）。

## 架构

代理使用插件架构：
- **oproxy/config.py**: 定义 LLM 提供商配置和 URL 路由规则
- **oproxy/plugins.py**: 拦截请求并路由到合适提供商的插件
- **main.py**: 使用 proxy.py CLI 的服务器入口点

## 环境配置 (使用 uv)

### 安装依赖
```bash
# 安装 uv (如果未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate  # Unix/macOS
# 或 .venv\Scripts\activate  # Windows

uv pip install -r requirements.txt
```

### 环境变量设置
```bash
# OpenAI
export OPENAI_API_KEY="your key"
export OPENAI_BASE_URL="your base rul"  # 可选

# Claude
export ANTHROPIC_API_KEY="your key"
export ANTHROPIC_BASE_URL="your base url"  # 可选

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your key"
export AZURE_OPENAI_BASE_URL="your base url"
```

## 使用命令

### 启动服务器
```bash
python main.py
```

### 测试配置
```bash
python tests/test_openai_direct.py       # OpenAI 验证
python tests/test_anthropic_direct.py    # Anthropic 验证
```

### URL 路由
- `http://localhost:8080/openai/v1/chat/completions` → OpenAI
- `http://localhost:8080/anthropic/v1/messages` → Anthropic
- `http://localhost:8080/azure/chat/completions` → Azure OpenAI


## 插件结构

LLMProxyPlugin 在 `oproxy/plugins.py` 中实现：
- `before_routing()`
- `routes()`
- `handle_route()`: 
- `_update_request_headers()`

## 配置

提供商配置在 `oproxy/config.py` 中定义，可通过环境变量自定义：
- `OPENAI_BASE_URL`: 覆盖 OpenAI 端点
- `ANTHROPIC_BASE_URL`: 覆盖 Claude 端点
- `AZURE_OPENAI_BASE_URL`: 设置 Azure 端点

所有提供商配置都包含特定于每个服务的 API 端点和认证头。