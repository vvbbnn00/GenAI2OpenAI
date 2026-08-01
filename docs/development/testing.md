# 测试与验证

## 离线测试

默认 pytest 不访问真实 GenAI，不读取 keystore：

```bash
UV_CACHE_DIR=/tmp/genai2openai-uv-cache \
GENAI_TOKENIZER_CACHE=/tmp/genai2openai-tokenizer-cache \
uv run pytest -q
```

模型官方资源测试使用 fixture 或受控缓存，不应把普通测试变成隐式联网测试。

常用静态检查：

```bash
UV_CACHE_DIR=/tmp/genai2openai-uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/genai2openai-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/tmp/genai2openai-uv-cache uv run python -m compileall -q src tests
UV_CACHE_DIR=/tmp/genai2openai-uv-cache uv lock --check
docker compose config --quiet
git diff --check
```

修改 Python 文件后，可用 `uv run ruff format .` 统一格式。格式化只处理排版，功能变化
仍应单独修改、测试和 review。

构建产物检查：

```bash
UV_CACHE_DIR=/tmp/genai2openai-uv-cache uv build
```

wheel 应只包含 `genai_proxy` 包，sdist 不应包含 keystore、`.env`、`.git`、tests 或
smoke 脚本。

## 测试目录

```text
tests/
├── api/          # 外部协议和事件形状
├── chat/         # 编排层边界
├── models/       # 官方 codec、token、工具解析
├── upstream/     # 认证、传输、缓存和 Kimi 清理
├── integration/  # 跨层请求流程
├── project/      # 包布局和版本元数据
└── live/         # 显式运行的真实上游测试
```

`scripts/smoke/` 是人工冒烟工具，不由 pytest 自动执行。

## 真实上游短矩阵

短矩阵覆盖 DeepSeek V4 Flash/Pro、GLM 5.2、Qwen 3.5 和 Kimi K3，包括普通响应、
reasoning、工具、视觉及传输副作用检查。必须显式提供 keystore：

```bash
UV_CACHE_DIR=/tmp/genai2openai-uv-cache \
GENAI_TOKENIZER_CACHE=/tmp/genai2openai-tokenizer-cache \
uv run python tests/live/allowed_models.py \
  --keystore /path/to/ids-passkey.keystore \
  --models deepseek-chat deepseek-pro chatglm qwen-instruct kimi-k3
```

runner 会审计最终聊天 payload，要求 `stream` 为 true、所有模型都没有
`chatGroupId`、Kimi 使用非空 `chatInfo`，并检查 Kimi 历史记录确实完成定位和删除。

这是会访问真实服务并产生模型用量的测试，不应放进普通 CI。

## OMP 长上下文工具链

OMP runner 只允许 GLM 5.2、DeepSeek V4 Flash 和 DeepSeek V4 Pro。默认生成至少
120,000 token 的上下文，执行至少 12 个依次依赖的工具阶段，并验证实际文件写入、
reasoning 增量、工具轮数和上下文标记：

```bash
UV_CACHE_DIR=/tmp/genai2openai-uv-cache \
GENAI_TOKENIZER_CACHE=/tmp/genai2openai-tokenizer-cache \
uv run python tests/live/omp_long_context.py \
  --keystore /path/to/ids-passkey.keystore \
  --omp-bin /path/to/omp \
  --models chatglm deepseek-chat deepseek-pro \
  --min-context-tokens 120000 \
  --stages 12 \
  --repeat 1
```

runner 在临时目录创建独立 OMP 配置和工作区，不修改用户现有 OMP 或 Claude Code
环境。一次通过只说明该次固定任务和上游状态成功，不等于所有长任务都不会失败。

## 人工冒烟

已有服务可用时：

```bash
uv run python scripts/smoke/tool_calling.py \
  --base-url http://127.0.0.1:5000/v1 \
  --model chatglm
```

提交前仍以自动测试为准，人工冒烟不能替代 parser、token 和 payload 单元测试。
