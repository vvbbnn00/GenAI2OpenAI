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
./scripts/docker-compose.sh up -d --build
```

4. 查看日志：

```bash
docker compose logs -f
```

启动日志会输出镜像对应的完整 Git commit hash 和提交时间，例如
`Program version: commit=<hash> committed_at=<ISO 8601> source=image`。
包装脚本只在宿主机读取 Git，然后把 hash、提交时间和脏工作区状态作为三个
短构建参数传给 Docker。`.git` 完全排除在构建上下文和镜像之外。
在 CI 或源码归档中也可以直接设置 `GENAI_BUILD_COMMIT`、
`GENAI_BUILD_COMMIT_TIME` 和 `GENAI_BUILD_DIRTY`。没有这些参数时，直接执行
`docker compose up -d --build` 仍能构建，但版本显示为
`Program version: local-dev`。
如果参与镜像运行的源码尚未提交，日志还会追加 `dirty=true`，避免把脏工作区
误认为该 commit 的完整内容。

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
| `GENAI_BUILD_COMMIT` | Docker 镜像对应的完整 Git commit hash，包装脚本自动设置 | 空 |
| `GENAI_BUILD_COMMIT_TIME` | Docker 镜像对应的 Git commit 时间，包装脚本自动设置 | 空 |
| `GENAI_BUILD_DIRTY` | 构建源码是否包含未提交改动，包装脚本自动设置 | `0` |
| `TOKEN_CHECK_INTERVAL` | 后台检查 token 过期时间并同步共享缓存的间隔秒数；`0` 表示关闭 | `60` |
| `GENAI_MAX_RETRIES` | GenAI 聊天、模型列表和计费请求遇到临时网络故障、HTTP 408/425/429/500/502/503/504、无效响应或意外断流时的最大重试次数；`0` 表示关闭 | `10` |
| `GENAI_RETRY_BACKOFF` | 首次重试等待秒数，后续每次翻倍并封顶 5 秒；`0` 表示立即重试 | `0.5` |
| `GENAI_MODEL_CACHE` | 最近一次有效 GenAI 模型表的持久化缓存文件 | `~/.cache/genai2openai/models.json` |
| `GENAI_TOKENIZER_CACHE` | 经 SHA-256 校验的官方 tokenizer/template 缓存目录 | `~/.cache/genai2openai/tokenizers` |
| `APP_PORT` | 容器内和映射到宿主机的监听端口 | `5000` |
| `HOST_PORT` | 宿主机暴露端口 | `5000` |
| `PROXY_API_KEY` | 代理自身的客户端认证密钥，会传给应用的 `API_KEY` 环境变量 | 空 |
| `APP_DEBUG` | 是否启用 `--debug`，`1` 为开启 | `0` |
| `CLAUDE_HAIKU_MODEL` | Claude haiku 别名映射到的 GenAI 模型 | `deepseek-chat` |
| `CLAUDE_SONNET_MODEL` | Claude sonnet 别名映射到的 GenAI 模型 | `chatglm` |
| `CLAUDE_OPUS_MODEL` | Claude opus 别名映射到的 GenAI 模型 | `chatglm` |
| `GUNICORN_WORKERS` | gunicorn worker 进程数 | `2` |
| `GUNICORN_THREADS` | 每个 worker 的线程数，决定长连接并发承载能力 | `8` |
| `GUNICORN_TIMEOUT` | 单请求超时，适合 LLM 长响应 | `180` |
| `GUNICORN_GRACEFUL_TIMEOUT` | 优雅退出超时 | `30` |
| `GUNICORN_KEEPALIVE` | HTTP keep-alive 秒数 | `10` |
| `GUNICORN_MAX_REQUESTS` | 单 worker 最多处理请求数，`0` 表示不轮换 | `0` |
| `GUNICORN_MAX_REQUESTS_JITTER` | worker 轮换抖动 | `0` |

`docker-compose.yml` 默认使用 `gunicorn + gthread`，比直接跑 Flask 开发服务器更适合本项目这种“请求会长时间阻塞在上游 LLM SSE 输出上”的代理场景。

模型表使用 stale-while-revalidate：内存缓存过期后，`/v1/models` 和模型解析会立即返回最后一次有效结果，并由单个后台线程刷新，不再让请求同步等待上游重试。成功获取的模型表会原子写入 `GENAI_MODEL_CACHE`，所以进程或容器重启后仍可使用；Docker Compose 默认把该文件放在独立持久卷中。上游暂时不可用、返回异常数据或 token 刷新失败时会继续提供缓存，冷启动且尚无缓存时则提供内置基础模型表，并按 30 秒冷却周期后台重试。

### 启动服务

```bash
uv run main.py --token <token> [--port 5000] [--api-key <key>] [--debug]
uv run main.py --keystore <path/to/ids-passkey.keystore> [--port 5000] [--api-key <key>] [--debug]
uv run main.py --token <token> --keystore <path/to/ids-passkey.keystore> [--port 5000] [--api-key <key>] [--debug]
uv run main.py --keystore <path/to/ids-passkey.keystore> --token-check-interval 60
uv run main.py --keystore <path/to/ids-passkey.keystore> --genai-max-retries 10 --genai-retry-backoff 0.5
uv run main.py --token <token> --claude-opus-model chatglm --claude-sonnet-model chatglm --claude-haiku-model deepseek-chat
```

直接运行源码时，启动日志中的版本信息来自当前仓库的最后一次提交，格式为
`Program version: commit=<hash> committed_at=<ISO 8601> source=git`。无法读取 Git
元数据时显示 `Program version: local-dev`。
源码有未提交改动时同样会追加 `dirty=true`。

认证参数说明：

- 必须提供 `--token` 或 `--keystore` 二者之一
- 只提供 `--token`：使用静态 token，不会自动刷新
- 只提供 `--keystore`：启动时自动通过 passkey 登录并获取 token，后续会在 token 即将过期时自动刷新
- 同时提供 `--token` 和 `--keystore`：优先使用现有 token 启动，并在后续自动刷新

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--token` | GenAI 平台的访问令牌（JWT）；和 `--keystore` 二选一或同时提供 | — |
| `--keystore` | `shanghaitech-ids-passkey` 生成的 keystore 文件路径，用于自动登录/刷新 token | — |
| `--token-check-interval` | 后台检查 token 过期时间并同步共享缓存的间隔秒数；`0` 表示关闭 | `60` |
| `--genai-max-retries` | GenAI 上游临时故障的最大重试次数；`0` 表示关闭 | `10` |
| `--genai-retry-backoff` | 首次重试等待秒数，后续每次翻倍并封顶 5 秒；`0` 表示立即重试 | `0.5` |
| `--port` | 服务监听端口 | `5000` |
| `--api-key` | 客户端认证密钥（也可通过 `API_KEY` 环境变量设置） | 无（不校验） |
| `--debug` | 启用详细日志输出 | 关闭 |
| `--claude-haiku-model` | 模型名包含 `haiku` 时映射到的 GenAI 模型，也可通过 `CLAUDE_HAIKU_MODEL` 环境变量设置 | `deepseek-chat` |
| `--claude-sonnet-model` | 模型名包含 `sonnet` 时映射到的 GenAI 模型，也可通过 `CLAUDE_SONNET_MODEL` 环境变量设置 | `chatglm` |
| `--claude-opus-model` | 模型名包含 `opus` 时映射到的 GenAI 模型，也可通过 `CLAUDE_OPUS_MODEL` 环境变量设置 | `chatglm` |

重试只会重放尚未交付给客户端的响应。非流式请求以及带工具请求的正文和工具调用会先收齐一轮上游响应，因此在客户端尚未收到内容时，上游即使返回了部分工具正文后断流，也可以安全地整轮重试。带工具的流式请求会立即转发独立的推理增量，但继续缓存正文和工具调用，直到完整解析成功。任何实时流一旦已经向客户端发送文本或推理内容，就不会从头重放，以免产生重复或互相矛盾的内容；此时代理会发送明确的流错误并结束请求。

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
- `POST /v1/responses` — OpenAI Responses API 兼容接口（流式/非流式）
- `GET /v1/models` — 列出可用模型
- `GET /v1/dashboard/billing/subscription` — 当前代理账号的订阅额度信息
- `GET /v1/dashboard/billing/usage` — 当前代理账号的使用量信息
- `POST /v1/messages` — Claude Messages API 兼容接口（流式/非流式）
- `POST /v1/messages/count_tokens` — Claude token 估算接口
- `GET /health` — 健康检查

### Tool Calling

支持 OpenAI 格式的 tool calling，通过模型专用 prompt 适配兼容部分不原生支持 function calling 的模型。Kimi-K3 例外，原因见下文。

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
| `MiniMax-M1` | MiniMax 2.7 | Xinference | 保留 MiniMax-M2.7 官方模板兼容代码；当前上游已移除该模型，不进入故障回退目录或在线测试 |
| `chatglm` | GLM 5.2；旧记录中也可能出现 GLM 5.1 | Xinference | GLM-5.2/5.1 官方 `<tools>`、`<tool_call>name<arg_key>...<arg_value>...` 提示与解析；5.2 使用上游模板自己的默认 `Reasoning Effort: Max` |
| `qwen-instruct` | Qwen 3.5 / Qwen3.5-397B-A17B | Xinference | Qwen3.5 官方 `<tool_call><function=...><parameter=...>` 提示、历史序列化与解析 |
| `kimi-k3` | Kimi-K3 | Xinference | 官方 `encoding_k3.py` 文本/视觉编码、视觉 token 计算和 XTML 解析；当前 GenAI 传输不支持官方工具声明 |

DeepSeek 同厂不同版本不会共用同一个 adapter：`deepseek-chat` 使用 `deepseek_v4_flash`，`deepseek-pro` 使用 `deepseek_v4_pro`，旧 DeepSeek 名称使用 `deepseek_legacy`。V4 初始工具提示词按 DeepSeek V4 Pro `encoding_dsv4.py` / `encoding/tests` 的 `## Tools`、`<｜DSML｜tool_calls>`、`### Available Tool Schemas` 结构生成；MiniMax 和 GLM 分别按官方 `chat_template.jinja` 的 `<minimax:tool_call>` 与 `<tool_call>...<arg_key>...` 结构生成。`chatglm` 没有明确版本线索时按 GLM-5.2 处理。

GLM-5.2 官方模板有 `high`、`max` 两档，但 GenAI 网页接口没有暴露该模板的 `reasoning_effort` 参数；因此实际可用的只有模板默认 `max`。代理把所有显式等级归一到 `max`，也不会再向 system 重复注入 `Reasoning Effort`，避免出现两个 `Max` 或 `Max`/`High` 冲突。DeepSeek V4 同样只有两档 effort，但 GenAI 另有明确的顶层 `thinking` 开关：`none` 和未传 reasoning 时发送 `thinking: false`；`minimal`、`low`、`medium`、`high` 映射到 `thinking: true` + `high`；`xhigh`、`max` 映射到 `thinking: true` + `max`。Claude Messages 的 `thinking.enabled`/`thinking.adaptive` 会开启该开关，未给 `output_config.effort` 时按 Anthropic 默认 `high`；`thinking.disabled` 会关闭它。只有 `max` 会把官方 `REASONING_EFFORT_MAX` 文本放在首条消息最前面。聊天请求仍不发送 `chatGroupId`。

Kimi-K3 始终启用 thinking。GenAI 网页通道没有独立的 thinking effort 字段，因此所有 OpenAI 推理等级都使用上游默认的 `max`，保证实际提示词与 token 计数一致。

GenAI 网页通道 `/htk/chat/start/chat` 不会把 Kimi-K3 的顶层 `tools/tool_choice`、消息内 `tools`、结构化工具历史或历史 `reasoning_content` 传给 Xinference，因此无法使用 Moonshot 官方 `type="tool-declare"` XTML 工具声明。代理在该限制下使用明确区分于原生 XTML 的客户端响应协议：把精简后的函数 schema 放在当前用户轮之前的一条普通 system 消息中，要求模型只返回两种正文信封之一。需要外部操作时返回含合法 JSON 的 `<k3_action>...</k3_action>`；任务确实完成时返回 `<k3_final>...</k3_final>`。代理完整收齐并校验信封、JSON 和工具名后，再转换为 OpenAI `tool_calls`、Responses `function_call` 或 Claude `tool_use`。`auto` 不会把普通 JSON 或无信封正文误执行为工具；无信封输出会触发有限次、强制动作的结构化重试。`required` 和指定函数在严格匹配已声明工具时仍可容错接收无信封 JSON。桥接消息不会进入 `chatInfo`，也不会生成 `Call-expression schemas`、`User request:` 或 `name(key=value)` 正文。Moonshot 官方 XTML 返回解析仍然保留，便于上游以后提供原生透传时直接使用。

该客户端响应协议不是 Moonshot 原生工具协议。普通 system/user/assistant 消息仍由固定 revision 的官方 `encoding_k3.py` 编码，工具 schema、结果记录和 continuation state 也按实际发送的普通消息正文计数。每个已完成调用被压成一条 `Completed client action result:` 用户消息，记录 `id`、工具名、参数和结果；不再把历史调用写成与当前动作相似的标签，避免长链中模型模仿历史标签。由于 GenAI 会丢弃历史 `reasoning_content`，代理只把最近一轮工具调用前的 reasoning 复制一次到普通 system continuation state，旧 reasoning 全部丢弃。这样既能让下一轮延续已有计划，也让 reasoning 状态开销不会随调用轮数累积。Responses 工具结果回合不会自动降级为 `tool_choice: "none"`，只有调用方显式指定时才会禁用动作。代理不根据任务关键词、工具名或模型措辞猜测是否继续，只校验动作、完成信封及调用方的 `tool_choice`。流式请求会立即把上游推理增量转换为 OpenAI `reasoning_content`、Responses `response.reasoning_text.delta` 或 Claude `thinking_delta`；动作和完成正文仍会等信封闭合并解析成功后再发送，不会把半截 JSON 或协议标签泄露给客户端。

Kimi-K3 还有一项 GenAI 通道限制：最后一条用户输入必须放在非空 `chatInfo`，否则上游会追加一个缺少 `content` 的用户消息并拒绝请求。GenAI 会把每个非空 `chatInfo` 写入历史记录，而其他模型可以把完整消息放在 `messages` 并保持 `chatInfo` 为空，所以该现象只在 Kimi-K3 上明显。代理从不在聊天请求中发送 `chatGroupId`；该分组 ID 由 GenAI 服务端生成。Kimi-K3 成功生成完整响应后，代理会在返回结束 chunk 前从历史接口精确找出本次新增的分组并删除。定位不到唯一记录、生成失败或删除接口异常时会安全跳过，不会误删其他记录，也不会破坏模型响应。工具 schema 和桥接指令只放在 `messages`，不会写入历史问题正文。

DeepSeek V4 Pro/Flash、GLM-5.2、GLM-5.1、`Qwen/Qwen3.5-397B-A17B` 和 MiniMax-M2.7 都使用各自官方 Hugging Face 仓库的固定 revision、`tokenizer.json` 与消息模板精确计数；DeepSeek 的 chat/thinking 序列会跟随实际发送的 `thinking` 布尔值，Qwen3.5 图像输入还会按同一 revision 的官方 resize、patch 和 merge 规则展开视觉占位 token。代理实际注入的非 K3 工具声明不是另外维护的手写副本，而是由固定 revision 的官方 `chat_template.jinja` 或 Python encoder 直接渲染。Qwen、GLM 和 MiniMax 保留标准 `assistant.tool_calls`、`tool` 与 reasoning 历史，交给上游模型模板编码。GenAI 会忽略 DeepSeek 的顶层结构化 `tools`，因此 DeepSeek V4 改由官方 `encoding_dsv4.py` 一次性渲染完整多轮 prompt，再无损装入 GenAI 支持的 system/user 消息外壳；差分测试同时覆盖 chat、thinking high、thinking max、历史 tool call 与 tool result，并要求桥接前后逐字节相等。只有官方模板未表达的 `tool_choice` 约束会作为普通 system 内容追加，整个结果仍由同一官方模板编码，并接受逐字一致性测试。Kimi-K3 使用固定 revision 的官方 `tiktoken.model` 与 `encoding_k3.py`，视觉 token 按官方 resize、patch 和 merge 规则计算。首次使用会下载并校验资源，之后复用本地缓存；无需安装 `transformers` 或 PyTorch。OpenAI `POST /v1/responses/input_tokens` 与 Anthropic `POST /v1/messages/count_tokens` 会走同一套模型解析和官方模板；Kimi-K3 工具请求会计算实际桥接 system 正文，生成用量按上游实际输出的 `<k3_action>` 文本计算，不按转换后的 XTML 或 OpenAI JSON 反推。没有官方 Hugging Face 仓库的闭源或非 Xinference 模型只能使用兼容估算，不能伪装成官方 tokenizer 计数。

上游 SSE 使用单字节增量读取，避免 `requests` 默认块大小延迟短 reasoning 片段；代理同时识别 `reasoning_content`、`reasoning` 和跨 chunk 的 `<think>...</think>`，并分别转换为 OpenAI `reasoning_content`、Responses 完整 reasoning item（added、content part、delta、done）与 Claude `thinking_delta`。Responses 事件携带一致的 `item_id`、`output_index`、`content_index` 和连续 `sequence_number`，严格客户端不需要猜测增量属于哪个输出项。SSE 响应设置 `X-Accel-Buffering: no` 和 `no-transform`，降低反向代理再次缓冲的风险。Responses 流式请求以及请求了 `stream_options.include_usage` 的 Chat Completions 都把精确 usage 计算延后到结束事件，不让 tokenizer 计算阻塞首个正文或 reasoning 增量。上游连续 90 秒没有任何字节时会按停滞处理，且只在尚未向客户端发送数据时重试一次；该窗口覆盖已实测超过 25 秒的冷启动，同时仍远短于旧的 600 秒超时和通用重试上限。

对于支持工具适配的模型，代理会优先保留 Claude Code 请求中 `Bash`、`Glob`、`Read`、`Edit`、`Write` 等精确工具名，并兼容模型偶发输出的 `Bash<arg_key>`、`<arg_value>`、`Bash\nIN\n...`、`Globpattern: ...`、DSML、MiniMax XML 和 JSON-ish 工具块。工具结果后的多轮会话会继续保留工具定义，允许模型按需继续调用工具；显式 `tool_choice: "none"` 时才禁止工具调用。

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

- 模型名包含 `haiku` -> 默认映射到 `deepseek-chat`（DeepSeek V4 Flash）
- 模型名包含 `sonnet` -> 默认映射到 `chatglm`（GLM-5.2）
- 模型名包含 `opus` -> 默认映射到 `chatglm`（GLM-5.2）

这些默认值都可以通过启动参数覆盖，例如：

```bash
uv run main.py \
  --keystore /path/to/ids-passkey.keystore \
  --claude-haiku-model deepseek-chat \
  --claude-sonnet-model chatglm \
  --claude-opus-model chatglm
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

`/v1/models` 会读取 GenAI 上游模型列表，返回当前账号在 GenAI 可见的模型。请求失败时会按统一策略重试；已有缓存过期但刷新仍失败时，会临时返回上一份缓存，避免模型列表短暂故障阻断聊天请求。它会自动带出上游的 `rootAiType`，并默认过滤 `gpt-image-1.5`。具体模型集合以上游返回为准。

## 测试

基础离线检查：

```bash
uv run --with pytest python -m pytest -q
uv run python -m compileall genai_proxy test_tool_adapters.py test_allowed_models_integration.py
uv run python test_tool_adapters.py
```

使用 `docker-deploy.keystore` 对 DeepSeek V4 Flash、DeepSeek V4 Pro、GLM-5.2、Qwen3.5、Kimi-K3 做 20 轮变体集成测试：

```bash
uv run python test_allowed_models_integration.py --repeat 20 --models deepseek-chat deepseek-pro chatglm qwen-instruct kimi-k3
```

该集成测试只允许调用 `deepseek-chat`、`deepseek-pro`、`chatglm`、`qwen-instruct`、`kimi-k3`。前四个模型覆盖 OpenAI 和 Claude Messages 两种调用风格下的非流式工具调用、流式工具调用、Claude Code 风格 `Bash` 工具、工具结果回合和无需工具的普通回答；支持 reasoning 流的模型还会验证 OpenAI `reasoning_content` 与 Responses 完整 reasoning item 事件链。Kimi-K3 覆盖 OpenAI/Claude 文本、OpenAI/Responses/Claude 视觉、token usage，以及外部操作桥接的 OpenAI 非流式和流式工具调用、Claude 流式工具调用和 Responses 长历史三轮工具链；三轮链路会验证空白续轮仍能流式调用下一项工具，最后再正常收敛。为限制 Kimi-K3 实时测试流量，其默认只执行 1 轮，不跟随通用 `--repeat`；确需重复时显式传入 `--kimi-repeat N`。成功完成的 Kimi-K3 测试历史会自动清理。

对 GLM-5.2、DeepSeek V4 Flash 和 DeepSeek V4 Pro 做隔离的 OMP 长上下文工具链测试：

```bash
uv run python test_omp_long_context_integration.py \
  --models chatglm deepseek-chat deepseek-pro \
  --min-context-tokens 150000 \
  --stages 12
```

该 runner 使用临时 OMP 配置、临时工作区和随机本地端口，不读取现有 OMP profile、规则、技能、扩展或项目 MCP 配置。它通过本地 keystore 启动代理，生成至少 15 万官方 tokenizer token 的上下文，并要求模型逐步读取 12 个带未知下一跳的文件。验证项包括首轮实际 input token、reasoning/tool-call/text 流增量、DeepSeek 每个工具回合的 reasoning 历史、全部阶段文件覆盖、工具执行错误、三个长上下文哨兵、最终 JSON 和完成标记。聊天请求仍不发送 `chatGroupId`。

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
4. 每隔 `TOKEN_CHECK_INTERVAL` 秒后台检查 token 过期时间并同步共享缓存
5. 在 token 即将过期前自动刷新，实际 API 请求被上游拒绝时也会立即重取

### 关于自动刷新

- 自动刷新基于 JWT 的 `exp` 字段判断过期时间
- 当前实现会在 **距离过期约 5 分钟** 时提前刷新
- 提供 `--keystore` 时默认每 60 秒后台检查一次 token 过期时间并同步共享缓存；可用 `--token-check-interval` 或 `TOKEN_CHECK_INTERVAL` 调整，设置为 `0` 可关闭
- 仅提供 `--token` 时，服务不会自动刷新，只会在日志中提示 token 即将过期
- 提供 `--keystore` 时，会保存更新后的 keystore（例如递增的 `sign_count`）

## 许可

MIT License — 详见 LICENSE 文件。
