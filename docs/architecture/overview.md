# 架构总览

项目采用 `src` 布局。distribution 名称是 `genai`，Python import 包名保持为
`genai_proxy`。

```text
.
├── main.py                    # 源码运行兼容入口
├── src/genai_proxy/
│   ├── api/                   # OpenAI 与 Anthropic 协议转换
│   ├── chat/                  # 请求准备、工具循环、流解析和 usage
│   ├── models/                # 各模型的官方 codec 与工具适配
│   ├── upstream/              # GenAI 认证、传输、目录和 Kimi 历史清理
│   ├── app.py                 # Flask 应用组装
│   ├── cli.py                 # 开发服务器 CLI
│   ├── runtime.py             # CLI/WSGI 共用启动逻辑
│   └── wsgi.py                # 生产 WSGI 入口
├── tests/                     # 按源码域镜像组织的离线和在线测试
├── scripts/smoke/             # 人工冒烟脚本
└── docs/                      # 本 Wiki
```

## 模块边界

`api` 负责外部协议，不处理模型格式。它把 OpenAI 或 Anthropic 请求转换为统一
参数，并把内部事件转换回客户端协议。

`chat` 是编排层。它负责消息规范化、模型解析、reasoning 配置、工具循环、SSE
增量、重试边界和 token usage。

`models` 只放模型家族知识，包括固定 Hugging Face 资源、官方模板调用、视觉 token
规则和工具输出解析。活动模型分别位于 `glm52/`、`deepseek_v4/`、`qwen35/` 和
`kimi_k3/`。

`upstream` 负责 GenAI 服务本身，包括 token 刷新、HTTP/SSE、模型目录缓存和 Kimi
历史清理。模型适配不直接发网络请求。

依赖方向保持为：

```text
api -> chat -> models
            -> upstream
```

`app.py` 只组装这些对象。`compat/`、`routes/`、`services/` 和 `optimizations/` 是旧
import 路径的薄兼容层，新的内部代码不依赖它们。

## 入口

- `main.py`、`python -m genai_proxy`、`genai2openai`：本地开发服务器
- `genai_proxy.wsgi:app`：Gunicorn 等生产 WSGI 服务器
- `create_app(config, logger)`：测试或嵌入时的应用工厂

CLI 和 WSGI 都经过 `runtime.py`，所以版本日志、环境变量和应用组装不会分叉。

## 为什么保留 `genai_proxy`

这个名称已经是项目对内 import 契约。重构只调整目录和职责，不改变包名，避免
破坏现有脚本、部署入口和第三方导入。根目录无需再增加同名业务包；`src` 布局已经
把可安装代码与测试、文档和部署文件分开。
