# GenAI2OpenAI

OpenAI 兼容的代理服务，将上海科技大学 GenAI 平台的 API 转换为标准 OpenAI Chat Completion 接口，支持 tool calling。

## 安装与运行

### 环境要求

- Python 3.11+
- 推荐使用 [uv](https://github.com/astral-sh/uv) 管理环境

### 安装依赖

```bash
uv sync
```

### 启动服务

```bash
uv run main.py --token <token> [--port 5000] [--api-key <key>] [--debug]
uv run main.py --keystore <path/to/ids-passkey.keystore> [--port 5000] [--api-key <key>] [--debug]
uv run main.py --token <token> --keystore <path/to/ids-passkey.keystore> [--port 5000] [--api-key <key>] [--debug]
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

设置 `--api-key` 或环境变量 `API_KEY` 后，所有 `/v1/` 请求需要携带 `Authorization: Bearer <key>` 请求头。未设置时跳过认证（开发模式）。

### 支持的模型

| 模型 | 后端类型 |
|------|----------|
| `GPT-5` | Azure |
| `GPT-4.1` | Azure |
| `GPT-4.1-mini` | Azure |
| `o4-mini` | Azure |
| `o3` | Azure |
| `deepseek-v3:671b` | Xinference |
| `deepseek-r1:671b` | Xinference |
| `qwen-instruct` | Xinference |
| `qwen-think` | Xinference |

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
