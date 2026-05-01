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

- 把本地 keystore 放到 `KEYSTORE_HOST_PATH` 指向的位置，默认是仓库根目录的 `./docker-deploy.keystore`
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
| `KEYSTORE_HOST_PATH` | 宿主机上的 keystore 文件路径，会挂载到容器内 | `./docker-deploy.keystore` |
| `TOKEN_CHECK_INTERVAL` | 后台确认 GenAI token 是否仍有效的间隔秒数；`0` 表示关闭 | `60` |
| `APP_PORT` | 容器内和映射到宿主机的监听端口 | `5000` |
| `HOST_PORT` | 宿主机暴露端口 | `5000` |
| `PROXY_API_KEY` | 代理自身的客户端认证密钥，会传给应用的 `API_KEY` 环境变量 | 空 |
| `APP_DEBUG` | 是否启用 `--debug`，`1` 为开启 | `0` |
| `CLAUDE_HAIKU_MODEL` | Claude haiku 别名映射到的 GenAI 模型 | `qwen-instruct` |
| `CLAUDE_SONNET_MODEL` | Claude sonnet 别名映射到的 GenAI 模型 | `qwen-instruct` |
| `CLAUDE_OPUS_MODEL` | Claude opus 别名映射到的 GenAI 模型 | `deepseek-v3:671b` |
| `GUNICORN_WORKERS` | gunicorn worker 进程数 | `2` |
| `GUNICORN_THREADS` | 每个 worker 的线程数，决定长连接并发承载能力 | `8` |
| `GUNICORN_TIMEOUT` | 单请求超时，适合 LLM 长响应 | `180` |
| `GUNICORN_GRACEFUL_TIMEOUT` | 优雅退出超时 | `30` |
| `GUNICORN_KEEPALIVE` | HTTP keep-alive 秒数 | `10` |
| `GUNICORN_MAX_REQUESTS` | 单 worker 最多处理请求数，`0` 表示不轮换 | `0` |
| `GUNICORN_MAX_REQUESTS_JITTER` | worker 轮换抖动 | `0` |

`docker-compose.yml` 默认使用 `gunicorn + gthread`，比直接跑 Flask 开发服务器更适合本项目这种“请求会长时间阻塞在上游 LLM SSE 输出上”的代理场景。

### 启动服务

```bash
uv run main.py --token <token> [--port 5000] [--api-key <key>] [--debug]
uv run main.py --keystore <path/to/ids-passkey.keystore> [--port 5000] [--api-key <key>] [--debug]
uv run main.py --token <token> --keystore <path/to/ids-passkey.keystore> [--port 5000] [--api-key <key>] [--debug]
uv run main.py --keystore <path/to/ids-passkey.keystore> --token-check-interval 60
uv run main.py --token <token> --claude-opus-model deepseek-chat --claude-sonnet-model MiniMax-M1 --claude-haiku-model chatglm
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
| `--token-check-interval` | 后台确认 GenAI token 是否仍有效的间隔秒数；`0` 表示关闭 | `60` |
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
    model="deepseek-chat",
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

### Xinference 模型工具适配

GenAI 的模型名并不总是直接等于上游模型名，代理会结合 `/v1/models` 返回的 `aiType`、`aiName`、`simpleName`、`descInfo`、`rootAiType`、`rootModelName` 判断具体适配器。`rootModelName` 明确不是 `Xinference` 时不会套用这些 Xinference 专用适配。

当前对以下 Xinference 上游模型有专用 tool calling 适配：

| GenAI 模型名 | GenAI 描述线索 | 上游 | 适配重点 |
|-------------|----------------|------|----------|
| `deepseek-chat` | DeepSeek V4 Flash / DeepSeek V4 | Xinference | DeepSeek V4 官方 DSML `<｜DSML｜tool_calls>` 注入与解析，兼容旧 `<｜DSML｜function_calls>` |
| `deepseek-pro` | DeepSeek V4 Pro / DeepSeek V4 | Xinference | DeepSeek V4 官方 DSML `<｜DSML｜tool_calls>` 注入与解析，兼容旧 `<｜DSML｜function_calls>` |
| `MiniMax-M1` | MiniMax 2.7 | Xinference | MiniMax-M2.7 官方 `<minimax:tool_call>` 注入与解析，兼容 XML/JSON-ish 变体和 `<think>` 过滤 |
| `chatglm` | GLM 5.1 | Xinference | GLM-5.1 官方 `<tool_call>name<arg_key>...<arg_value>...` 注入与解析，兼容非标准闭合标签 |

DeepSeek 同厂不同版本不会共用同一个 adapter：`deepseek-chat` 使用 `deepseek_v4_flash`，`deepseek-pro` 使用 `deepseek_v4_pro`，旧 DeepSeek 名称使用 `deepseek_legacy`。V4 初始工具提示词按 DeepSeek V4 Pro `encoding_dsv4.py` / `encoding/tests` 的 `## Tools`、`<｜DSML｜tool_calls>`、`### Available Tool Schemas` 结构生成；MiniMax 和 GLM 分别按官方 `chat_template.jinja` 的 `<minimax:tool_call>` 与 `<tool_call>...<arg_key>...` 结构生成。

GenAI 网页通道 `/htk/chat/start/chat` 目前不暴露可靠的原生 `tools/tool_choice` 通道，因此代理不会向上游请求体拼接 `tools` 或 `tool_choice` 字段，而是把工具定义写入模型专用的隐藏提示词。返回时统一解析成 OpenAI 或 Claude Messages API 的结构化工具调用响应，不把模型生成的 `<tool_call>`、`<minimax:tool_call>`、DSML 或 Claude Code transcript 片段直接透传给客户端。

针对 Claude Code 常见的 `Bash`、`Glob`、`Read`、`Edit`、`Write` 等工具，代理会优先保留请求里的精确工具名，并兼容模型偶发输出的 `Bash<arg_key>`、`<arg_value>`、`Bash\nIN\n...`、`Globpattern: ...`、DSML、MiniMax XML 和 JSON-ish 工具块。工具结果后的多轮会话会继续保留工具定义，允许模型按需继续调用工具；显式 `tool_choice: "none"` 时才禁止工具调用。

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
  --claude-haiku-model chatglm \
  --claude-sonnet-model MiniMax-M1 \
  --claude-opus-model deepseek-chat
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

## 测试

基础离线检查：

```bash
uv run python -m compileall genai_proxy test_tool_adapters.py test_allowed_models_integration.py
uv run python test_tool_adapters.py
```

使用 `docker-deploy.keystore` 对 DeepSeek V4 Flash、DeepSeek V4 Pro、MiniMax 2.7、GLM-5.1 做 20 轮变体集成测试：

```bash
uv run python test_allowed_models_integration.py --repeat 20 --models deepseek-chat deepseek-pro MiniMax-M1 chatglm
```

该集成测试只允许调用 `deepseek-chat`、`deepseek-pro`、`MiniMax-M1`、`chatglm`，覆盖 OpenAI 和 Claude Messages 两种调用风格下的非流式工具调用、流式工具调用、Claude Code 风格 `Bash` 工具、工具结果回合和无需工具的普通回答。

## 项目结构

项目按职责分为以下几层：

- `main.py`：负责参数解析、日志初始化和服务启动
- `genai_proxy/app.py`：应用装配
- `genai_proxy/auth.py`：API Key 鉴权
- `genai_proxy/services/token_manager.py`：JWT / passkey 刷新
- `genai_proxy/services/genai.py`：GenAI 上游调用与 OpenAI SSE 转换
- `genai_proxy/compat/openai.py`：OpenAI tool calling 兼容逻辑
- `genai_proxy/compat/claude.py`：Claude Messages API 转换逻辑
- `genai_proxy/optimizations/`：模型专用 tool calling 适配与解析逻辑
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
4. 每隔 `TOKEN_CHECK_INTERVAL` 秒后台确认 token 是否仍有效
5. 在 token 即将过期前自动刷新，或在后台确认发现 token 被上游拒绝时立即重取

### 关于自动刷新

- 自动刷新基于 JWT 的 `exp` 字段判断过期时间
- 当前实现会在 **距离过期约 5 分钟** 时提前刷新
- 提供 `--keystore` 时默认每 60 秒后台确认一次 token 是否仍被 GenAI 接受；可用 `--token-check-interval` 或 `TOKEN_CHECK_INTERVAL` 调整，设置为 `0` 可关闭
- 仅提供 `--token` 时，服务不会自动刷新，只会在日志中提示 token 即将过期
- 提供 `--keystore` 时，会保存更新后的 keystore（例如递增的 `sign_count`）

## 许可

MIT License — 详见 LICENSE 文件。
