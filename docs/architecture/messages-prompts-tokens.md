# 消息、prompt 与 token

## 统一消息表示

API 层先把三种外部协议转换为同一套消息。文本 `content` 可以是字符串、`null` 或
内容块数组。内部支持的内容块是：

- `text`
- `image`
- `image_url`

纯文本块按原顺序直接拼接，不额外插入换行。图片只能出现在 user 消息中，来源必须
是 `http`、`https` 或非空的 base64 image data URL。所有验证都发生在工具注入、
token 计数和网络请求之前，因此三个 API 的接受范围一致。

当前只有 Qwen 3.5 和 Kimi K3 adapter 接受视觉输入。其他模型收到图片会返回明确的
客户端错误，不会静默丢图。

## 官方 prompt 规则

对于有官方 Hugging Face 模板或 encoder 的活动模型，项目把它视为第一方 prompt
实现，不维护另一份“看起来相似”的手写模板：

| 模型 | 固定资源 | 用途 |
| --- | --- | --- |
| GLM 5.2 | `tokenizer.json`、`chat_template.jinja` | 消息边界、工具声明、token 计数 |
| DeepSeek V4 | `tokenizer.json`、`encoding/encoding_dsv4.py` | 完整多轮 prompt、DSML 工具、token 计数 |
| Qwen 3.5 | `tokenizer.json`、`chat_template.jinja` | 消息边界、工具声明、视觉占位、token 计数 |
| Kimi K3 | `tiktoken.model`、`encoding_k3.py` | 普通消息边界、XTML 解析、视觉和 token 计数 |

资源按完整 commit revision 下载并校验 SHA-256。缓存文件损坏或 hash 不符时不会继续
使用。固定 revision 和 hash 见各模型页面及源码 `src/genai_proxy/models/*/codec.py`。

system 与 developer 消息会按模型官方模板的角色能力处理。GLM、DeepSeek、Qwen 的
工具声明由固定官方模板或 encoder 渲染。只有官方模板没有表达的 `tool_choice`
限制才追加为普通 system 内容。

DeepSeek V4 是特殊但仍属第一方 prompt 的情况。GenAI 不透传其结构化 `tools`，所以
代理先用官方 `encoding_dsv4.py` 渲染完整多轮 prompt，再拆成 GenAI 能接受的
system/user 外壳，并验证重新编码后的字节序列与官方结果完全一致。

## Kimi 工具桥接的边界

GenAI 通道会丢弃 Kimi 的原生 message-level 工具声明和结构化工具历史，因此代理
不能声称自己发送了 Moonshot 官方 `tool-declare` XTML。需要工具时，项目发送一个
普通 system 消息，要求模型用 `<k3_action>` 或 `<k3_final>` 信封响应。这个协议是
项目自定义的通道桥接，不是 Kimi 官方 prompt 格式。

普通 Kimi 消息的编码、视觉规则和 token 计数仍与官方 `encoding_k3.py` 一致；桥接
system 文本也按它实际发送的普通消息内容计数。项目不会生成旧的
`Call-expression schemas (JSON):`、`User request:` 或 `name(key=value)` 格式。

## Token 计算

输入 token 在完成模型解析、消息规范化、reasoning 归一和工具 prompt 注入后计算，
因此计数对象与实际上游请求一致。输出 token 根据上游原始 completion 计算，不会用
转换后的 OpenAI 或 Anthropic JSON 反推。

Qwen 和 Kimi 图片先读取尺寸，再按各自官方 resize、patch 和 merge 规则计算视觉
token。图片字节不会作为普通 base64 文本计数。

没有可用官方公开 tokenizer 的闭源或非 Xinference 模型使用兼容估算。API 会继续
工作，但该结果不应解释为官方精确计数。
