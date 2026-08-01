# 安装与首次请求

## 环境要求

- Python 3.11 或更高版本
- [uv](https://docs.astral.sh/uv/)
- 可用的 GenAI JWT，或 `shanghaitech-ids-passkey` keystore

克隆仓库后安装锁定依赖：

```bash
uv sync --frozen
```

## 直接运行

静态 token 模式：

```bash
uv run main.py --token <genai-jwt>
```

keystore 自动登录和刷新模式：

```bash
uv run main.py --keystore /path/to/ids-passkey.keystore
```

也可以同时提供两者。程序先使用现有 token，后续通过 keystore 刷新：

```bash
uv run main.py \
  --token <genai-jwt> \
  --keystore /path/to/ids-passkey.keystore
```

`main.py` 是源码运行的兼容入口。`python -m genai_proxy` 和安装后的
`genai2openai` 命令执行同一套 CLI。

## Docker Compose

先复制配置模板：

```bash
cp .env.example .env
```

token 模式至少设置：

```env
GENAI_TOKEN=<genai-jwt>
APP_PORT=5000
```

keystore 模式需要同时设置容器内路径和宿主机路径：

```env
KEYSTORE_PATH=/app/docker-deploy.keystore
KEYSTORE_HOST_PATH=/path/to/ids-passkey.keystore
```

keystore 以可写方式挂载，因为 passkey 计数器会更新。不要把 keystore 复制进
镜像，也不要提交 `.env`。

构建并启动：

```bash
./scripts/docker-compose.sh up -d --build
docker compose logs -f
```

包装脚本只把 commit hash、提交时间和脏工作区标记作为构建参数传入。`.git` 不在
Docker 构建上下文或镜像中。

## 发出第一个请求

Chat Completions 流式请求：

```bash
curl http://127.0.0.1:5000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "chatglm",
    "messages": [{"role": "user", "content": "用一句话介绍上海"}],
    "stream": true
  }'
```

OpenAI Python SDK：

以下代码在客户端运行，需要客户端环境自行安装 `openai`；代理服务本身不依赖该 SDK。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5000/v1",
    api_key="<proxy-key>",
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好"}],
)
print(response.choices[0].message.content)
```

未配置代理 API key 时，SDK 仍要求一个非空字符串，但服务端不会校验它。

## Hugging Face 资源

GLM、DeepSeek、Qwen 和 Kimi 的精确 token 计数依赖固定 revision 的官方
Hugging Face 资源。直接运行源码时，第一次使用对应模型会下载资源、校验 SHA-256
后写入 `GENAI_TOKENIZER_CACHE`。

Docker build 会预先下载当前维护模型的全部资源，并实际加载 tokenizer、chat template
和 Python encoder。任何资源下载、hash 或加载错误都会使构建失败。镜像运行时使用只读
缓存和离线模式，因此容器启动及首次请求都不会再访问 Hugging Face。

下一步可阅读 [配置参考](operations/configuration.md) 和
[模型适配总览](models/index.md)。
