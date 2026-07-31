# Qwen 3.5

## 识别与官方资源

GenAI ID 通常为 `qwen-instruct`，实际模型由目录记录识别为 Qwen 3.5。

- 仓库：[Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
- 固定 revision：`8472618112abcbd45acbcdc58436aff4233c23f7`
- `tokenizer.json` SHA-256：
  `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
- `chat_template.jinja` SHA-256：
  `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`

消息边界、工具声明、历史工具调用和 token 计数都使用这个固定模板。项目不维护另一份
手写的 Qwen 工具 prompt。

## Tool calling

代理解析官方格式：

```text
<tool_call>
<function=name>
<parameter=key>
value
</parameter>
</function>
</tool_call>
```

工具名和参数会在返回客户端前校验。`tool_choice` 的 required、none 和指定函数限制
会附加到官方渲染结果中，不改变工具 schema 的第一方模板来源。

结构化 `assistant.tool_calls` 和 `tool` 历史会保留到后续轮次，适合连续工具任务。

## 视觉输入

Qwen 3.5 接受 URL 或 base64 data URL 图片。代理读取实际宽高，按官方规则计算：

- patch size：16
- merge size：2
- 最小像素：65,536
- 最大像素：16,777,216
- 最大宽高比：200

宽高比或尺寸不合法时在请求上游前返回错误。视觉 token 使用 resize 后的 patch grid，
不会把 base64 字符串当普通文本。

## Reasoning 与流式输出

官方模板以 `enable_thinking=True` 渲染。当前 GenAI 通道没有 Qwen 专用顶层 thinking
开关，因此代理不会伪造一个上游没有的布尔字段，也不承诺客户端 effort 对应某个
Qwen 内部档位。

reasoning 与正文按 SSE 增量解析。工具语法仍在闭合并验证前缓冲。
