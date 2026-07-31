# 配置参考

## 认证上游

程序必须取得 GenAI JWT。可使用静态 token、keystore，或同时使用：

| CLI | 环境变量 | 说明 |
| --- | --- | --- |
| `--token` | `GENAI_TOKEN` | 现有 JWT |
| `--keystore` | `KEYSTORE_PATH` | passkey keystore，用于登录和刷新 |
| `--token-check-interval` | `TOKEN_CHECK_INTERVAL` | token 检查间隔秒数，`0` 关闭，默认 `60` |

只提供 token 时不会自动刷新。提供 keystore 时，认证失败可触发一次刷新，后台线程也会
在 token 接近过期时维护它。keystore 包含敏感认证材料，不能提交或写入镜像。

## 客户端认证

CLI 的 `--api-key` 或环境变量 `API_KEY` 用于保护代理。Docker Compose 对外使用
`PROXY_API_KEY`，再映射到容器内 `API_KEY`。

未设置时服务处于开放模式。OpenAI 接口使用 Bearer token；Anthropic Messages 还
接受 `x-api-key`。`/health` 不需要认证。

## 服务参数

| CLI | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--port` | `APP_PORT` | `5000` | 监听端口 |
| `--debug` | `APP_DEBUG=1` | 关闭 | 增加结构和长度等诊断元数据，不记录请求正文，也不启用 Flask reloader |
| `--genai-max-retries` | `GENAI_MAX_RETRIES` | `10` | 临时上游错误最大重试次数 |
| `--genai-retry-backoff` | `GENAI_RETRY_BACKOFF` | `0.5` | 首次退避秒数，翻倍后最高 5 秒 |
| `--genai-model-cache` | `GENAI_MODEL_CACHE` | `~/.cache/genai2openai/models.json` | 持久模型目录缓存 |

`GENAI_TOKENIZER_CACHE` 没有对应 CLI 参数。它指定经 hash 校验的 Hugging Face
资源缓存目录，默认使用用户缓存目录。

重试次数不是流式重放许可。只要客户端已收到任何增量，本轮就不会从头重试，详情见
[可靠性与失败边界](reliability.md)。

## Claude 别名

| CLI | 环境变量 | 默认值 |
| --- | --- | --- |
| `--claude-haiku-model` | `CLAUDE_HAIKU_MODEL` | `deepseek-chat` |
| `--claude-sonnet-model` | `CLAUDE_SONNET_MODEL` | `chatglm` |
| `--claude-opus-model` | `CLAUDE_OPUS_MODEL` | `chatglm` |

映射按请求模型名中出现的 tier 选择。请只映射到当前 `/v1/models` 可用且已经验证的
GenAI ID。

## Docker/Gunicorn

Docker Compose 还支持：

| 变量 | 默认值 |
| --- | --- |
| `HOST_PORT` | `5000` |
| `KEYSTORE_HOST_PATH` | `./docker-deploy.keystore` |
| `GUNICORN_WORKERS` | `2` |
| `GUNICORN_THREADS` | `8` |
| `GUNICORN_TIMEOUT` | `180` |
| `GUNICORN_GRACEFUL_TIMEOUT` | `30` |
| `GUNICORN_KEEPALIVE` | `10` |
| `GUNICORN_MAX_REQUESTS` | `0` |
| `GUNICORN_MAX_REQUESTS_JITTER` | `0` |

模型请求会长期占用上游 SSE 连接。线程数决定单个 worker 可同时承载的阻塞连接数；
调整时同时考虑内存、文件描述符和反向代理超时。

## 构建版本参数

`GENAI_BUILD_COMMIT`、`GENAI_BUILD_COMMIT_TIME` 和 `GENAI_BUILD_DIRTY` 只用于镜像
构建元数据。正常情况下应通过 `scripts/docker-compose.sh` 自动设置，不要手写一个与
源码不符的 hash。
