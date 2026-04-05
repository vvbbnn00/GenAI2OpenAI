# GenAI2OpenAI

OpenAI 兼容的代理服务，将上海科技大学 GenAI 平台的 API 转换为标准 OpenAI Chat Completion 接口，支持 tool calling。

现在也提供 Claude Messages API 兼容入口，可以让 Anthropic / Claude SDK 通过同一个代理访问 GenAI。

## 安装与运行

### 环境要求

- Python 3.11+
- 推荐使用 [uv](https://github.com/astral-sh/uv) 管理环境

### 安装依赖

```bash
uv sync
```

### 使用 Docker Compose 启动

仓库已内置 `Dockerfile`、`docker-compose.yml` 和 `.env.example`。

1. 复制环境变量模板：

```bash
cp .env.example .env
```

2. 在 `.env` 中至少配置 `GENAI_TOKEN` 或 `KEYSTORE_PATH` 其中一个。

基于 token 启动的最小配置：

```env
GENAI_TOKEN=<genai-jwt>
APP_PORT=5000
PROXY_API_KEY=
APP_DEBUG=0
```

如果你要使用 keystore 自动登录/自动刷新：

- 将 `docker-compose.yml` 里的 `volumes` 注释取消
- 把本地 keystore 挂载到容器内，例如 `/app/docker-deploy.keystore`
- 在 `.env` 中设置 `KEYSTORE_PATH=/app/docker-deploy.keystore`

3. 构建并启动服务：

```bash
docker compose up -d --build
```

4. 查看日志：

```bash
docker compose logs -f
```

5. 停止服务：

```bash
docker compose down
```

`docker-compose.yml` 里支持的主要环境变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `GENAI_TOKEN` | GenAI 平台 JWT；和 `KEYSTORE_PATH` 二选一或同时提供 | 空 |
| `KEYSTORE_PATH` | 容器内的 keystore 路径，用于 passkey 自动登录/刷新 | 空 |
| `APP_PORT` | 容器内和映射到宿主机的监听端口 | `5000` |
| `PROXY_API_KEY` | 代理自身的客户端认证密钥，会传给应用的 `API_KEY` 环境变量 | 空 |
| `APP_DEBUG` | 是否启用 `--debug`，`1` 为开启 | `0` |
| `CLAUDE_HAIKU_MODEL` | Claude haiku 别名映射到的 GenAI 模型 | `qwen-instruct` |
| `CLAUDE_SONNET_MODEL` | Claude sonnet 别名映射到的 GenAI 模型 | `qwen-instruct` |
| `CLAUDE_OPUS_MODEL` | Claude opus 别名映射到的 GenAI 模型 | `deepseek-v3:671b` |

### 启动服务

```bash
uv run main.py --token <token> [--port 5000] [--api-key <key>] [--debug]
uv run main.py --keystore <path/to/ids-passkey.keystore> [--port 5000] [--api-key <key>] [--debug]
uv run main.py --token <token> --keystore <path/to/ids-passkey.keystore> [--port 5000] [--api-key <key>] [--debug]
uv run main.py --token <token> --claude-opus-model deepseek-v3:671b --claude-sonnet-model qwen-instruct --claude-haiku-model qwen-instruct
```

认证参数说明：

- 必须提供 `--token` 或 `--keystore` 二者之一
- 只提供 `--token`：使用静态 token，不会自动刷新
- 只提供 `--keystore`：启动时自动通过 passkey 登录并获取 token，后续会在 token 即将过期时自动刷新
- 同时提供 `--token` 和 `--keystore`：优先使用现有 token 启动，并在后续自动刷新

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--token` | GenAI 平台的访问令牌（JWT）；和 `--keystore` 二选一或同时提供 | — |
| `--keystore` | `shanghaitech-ids-passkey` 生成的 keystore 文件路径，用于自动登录/刷新 token | — |
| `--port` | 服务监听端口 | `5000` |
| `--api-key` | 客户端认证密钥（也可通过 `API_KEY` 环境变量设置） | 无（不校验） |
| `--debug` | 启用详细日志输出 | 关闭 |
| `--claude-haiku-model` | 模型名包含 `haiku` 时映射到的 GenAI 模型，也可通过 `CLAUDE_HAIKU_MODEL` 环境变量设置 | `qwen-instruct` |
| `--claude-sonnet-model` | 模型名包含 `sonnet` 时映射到的 GenAI 模型，也可通过 `CLAUDE_SONNET_MODEL` 环境变量设置 | `qwen-instruct` |
| `--claude-opus-model` | 模型名包含 `opus` 时映射到的 GenAI 模型，也可通过 `CLAUDE_OPUS_MODEL` 环境变量设置 | `deepseek-v3:671b` |

### 启动示例

#### 1. 仅使用已有 token

```bash
uv run main.py --token <genai-jwt>
```

#### 2. 使用 passkey 自动登录与自动刷新

```bash
uv run main.py --keystore /path/to/ids-passkey.keystore
```

#### 3. 用现有 token 启动，并允许后续自动刷新

```bash
uv run main.py \
  --token <genai-jwt> \
  --keystore /path/to/ids-passkey.keystore
```

## 功能

### OpenAI 兼容接口

- `POST /v1/chat/completions` — 聊天补全（流式/非流式）
- `GET /v1/models` — 列出可用模型
- `GET /v1/dashboard/billing/subscription` — 当前代理账号的订阅额度信息
- `GET /v1/dashboard/billing/usage` — 当前代理账号的使用量信息
- `POST /v1/messages` — Claude Messages API 兼容接口（流式/非流式）
- `POST /v1/messages/count_tokens` — Claude token 估算接口
- `GET /health` — 健康检查

### Tool Calling

支持 OpenAI 格式的 tool calling，通过 prompt 注入实现，兼容不原生支持 function calling 的模型。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/v1",
    api_key="your-api-key"  # 如果设置了 --api-key
)

response = client.chat.completions.create(
    model="GPT-4.1",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    }]
)
```

支持 `tool_choice` 参数：`"auto"`（默认）、`"required"`、指定函数名。

### API Key 认证

设置 `--api-key` 或环境变量 `API_KEY` 后：

- OpenAI 兼容接口使用 `Authorization: Bearer <key>`
- Claude 兼容接口支持 `x-api-key: <key>` 或 `Authorization: Bearer <key>`

未设置时跳过认证（开发模式）。

计费接口 `/v1/dashboard/billing/*` 使用和其他 `/v1/*` 接口相同的代理鉴权方式。

它返回的是当前代理绑定的 GenAI 账号额度，而不是调用方自带用户身份的额度。

### Claude Messages API 兼容

除了 OpenAI 兼容接口外，本项目还提供了 Claude Messages API 兼容层：

- Claude 请求会先转换成内部统一的 OpenAI Chat Completion 请求
- 再复用原来的 GenAI 上游调用逻辑
- 返回时再转换回 Claude Messages API 的响应格式

当前支持的 Claude 能力：

- `messages`
- `system`
- `tools`
- `tool_choice`
- `tool_result`
- 流式 `tool_use`
- `count_tokens` 的简单估算

目前 Claude 路由使用关键词匹配来做 alias 映射：

- 模型名包含 `haiku` -> 默认映射到 `qwen-instruct`
- 模型名包含 `sonnet` -> 默认映射到 `qwen-instruct`
- 模型名包含 `opus` -> 默认映射到 `deepseek-v3:671b`

这些默认值都可以通过启动参数覆盖，例如：

```bash
uv run main.py \
  --keystore /path/to/ids-passkey.keystore \
  --claude-haiku-model qwen-instruct \
  --claude-sonnet-model GPT-4.1 \
  --claude-opus-model GPT-5.4
```

例如下面这些模型名都可以工作，只要名称中带有 `haiku`、`sonnet` 或 `opus`：

- `claude-3-5-haiku-latest`
- `claude-3-7-sonnet-latest`
- `claude-sonnet-4-0`
- `claude-opus-4-1`

Claude SDK 示例：

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:5000",
    api_key="your-api-key",
)

resp = client.messages.create(
    model="claude-3-7-sonnet-latest",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "你好，帮我总结一下牛顿第一定律"}
    ],
)

print(resp)
```

### 支持的模型

`/v1/models` 会实时读取 GenAI 上游模型列表，返回当前账号在 GenAI 可见的模型。它会自动带出上游的 `rootAiType`，并默认过滤 `gpt-image-1.5`。具体模型集合以 `/v1/models` 的实时返回为准。

## 项目结构

项目按职责分为以下几层：

- `main.py`：负责参数解析、日志初始化和服务启动
- `genai_proxy/app.py`：应用装配
- `genai_proxy/auth.py`：API Key 鉴权
- `genai_proxy/services/token_manager.py`：JWT / passkey 刷新
- `genai_proxy/services/genai.py`：GenAI 上游调用与 OpenAI SSE 转换
- `genai_proxy/compat/openai.py`：OpenAI tool calling 兼容逻辑
- `genai_proxy/compat/claude.py`：Claude Messages API 转换逻辑
- `genai_proxy/routes/`：OpenAI / Claude 路由

## Token 与 Passkey

### 方式一：手动获取 token

1. 前往 [GenAI 对话平台](https://genai.shanghaitech.edu.cn/dialogue)
2. 打开浏览器开发者工具，发送一条消息，捕获 `chat` 请求
3. 复制请求头中的 `x-access-token` 字段

![Token 获取示意](images/chrome.png)

这种方式适合临时使用，但 token 过期后需要你手动重新获取。

### 方式二：使用 passkey 自动登录

本项目已支持通过 `shanghaitech-ids-passkey` 自动登录上海科技大学 IDS，并通过 GenAI 登录流程自动拿到新的 JWT token。

如果你还没有 keystore，可以先使用 `shanghaitech-ids-passkey` 项目完成 passkey 绑定并生成 `.keystore` 文件，再在本项目中通过 `--keystore` 引用它。

使用前你需要先准备好一个 passkey keystore 文件，例如：

```bash
uv run main.py --keystore /path/to/ids-passkey.keystore
```

如果你已经有 keystore，则服务会：

1. 启动时通过 passkey 登录 IDS
2. 自动访问 GenAI 登录入口 `https://genai.shanghaitech.edu.cn/htk/user/login`
3. 从最终跳转 URL 中提取 `?token=...`
4. 在 token 即将过期前自动刷新

### 关于自动刷新

- 自动刷新基于 JWT 的 `exp` 字段判断过期时间
- 当前实现会在 **距离过期约 5 分钟** 时提前刷新
- 仅提供 `--token` 时，服务不会自动刷新，只会在日志中提示 token 即将过期
- 提供 `--keystore` 时，会保存更新后的 keystore（例如递增的 `sign_count`）

## 许可

MIT License — 详见 LICENSE 文件。
