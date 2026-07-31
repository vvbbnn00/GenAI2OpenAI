# Anthropic Messages

接口：

- `POST /v1/messages`
- `POST /v1/messages/count_tokens`

## 基本请求

```json
{
  "model": "claude-sonnet-4",
  "max_tokens": 2048,
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "stream": true
}
```

Claude 模型名按名称中的 tier 映射到 GenAI 模型。默认值：

| tier | GenAI 模型 |
| --- | --- |
| haiku | `deepseek-chat` |
| sonnet | `chatglm` |
| opus | `chatglm` |

可通过 CLI 或 `CLAUDE_HAIKU_MODEL`、`CLAUDE_SONNET_MODEL`、
`CLAUDE_OPUS_MODEL` 修改。这里是兼容映射，不表示上游实际运行 Claude。

## 内容块与工具

支持文本、图片、`tool_use` 和 `tool_result` 内容块。Anthropic 工具声明会转换为
OpenAI function tool，模型返回的调用再转换回 `tool_use`。

指定工具的格式是：

```json
{"tool_choice": {"type": "tool", "name": "get_weather"}}
```

工具结果必须保留对应 `tool_use_id`。消息转换与 Chat Completions 共用同一套校验，
因此不支持的图片模型和非法 URL 会在访问上游前返回错误。

## Thinking 与 effort

Anthropic 请求支持：

```json
{
  "thinking": {"type": "enabled"},
  "output_config": {"effort": "high"}
}
```

`thinking.type` 接受 `enabled`、`adaptive` 和 `disabled`。`output_config.effort`
接受 `low`、`medium`、`high`、`xhigh` 和 `max`。enabled/adaptive 未指定 effort 时
使用 `high`；disabled 映射为 `none`。具体模型仍会按自身可用档位归一。

## 流式响应

流式响应使用 Anthropic SSE 事件。上游 reasoning 转换为 `thinking` 内容块和
`thinking_delta`，普通正文转换为 `text_delta`，工具参数转换为
`input_json_delta`。内容块严格按 start、delta、stop 顺序发送。

## Token 计数

`POST /v1/messages/count_tokens` 接受与 Messages 相同的 model、system、messages、
tools、tool_choice 和 thinking 配置，返回：

```json
{"input_tokens": 8}
```

数值示例只说明响应形状。该接口与实际生成共用模型解析和 prompt 组装路径。

## 认证

如果代理配置了 API key，Messages 接口接受 `x-api-key: <proxy-key>`，也接受
`Authorization: Bearer <proxy-key>`。
