# OpenAI Chat Completions

接口：`POST /v1/chat/completions`

## 基本请求

```json
{
  "model": "deepseek-chat",
  "messages": [
    {"role": "system", "content": "回答要简洁。"},
    {"role": "user", "content": "解释什么是 SSE。"}
  ],
  "stream": true,
  "reasoning": {"effort": "high"}
}
```

`model` 缺省时使用 `GPT-4.1`。`messages` 必须是数组，消息与图片内容的统一规则见
[消息、prompt 与 token](../architecture/messages-prompts-tokens.md)。

OpenAI reasoning effort 接受 `none`、`minimal`、`low`、`medium`、`high`、
`xhigh` 和 `max`。模型可能只支持其中一部分实际档位，归一规则见对应模型页面。

## Tool calling

仅支持 `type: "function"` 工具。示例：

```json
{
  "model": "chatglm",
  "messages": [
    {"role": "user", "content": "查询上海天气"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

支持的 `tool_choice`：

- `auto`
- `required`
- `none`
- `{"type":"function","function":{"name":"get_weather"}}`

返回 `tool_calls` 后，客户端应执行工具，再把结果作为 `role: "tool"` 消息发回，并
保留本轮仍可使用的工具定义。代理不会执行工具。

## 流式响应

`stream: true` 返回 SSE。推理文本位于 `choices[0].delta.reasoning_content`，普通
正文位于 `choices[0].delta.content`。上游也可能使用 `reasoning` 或 `<think>` 标签，
代理会统一解析为 `reasoning_content`。

无工具请求的推理和正文会尽快转发。工具请求中的推理仍实时转发，但候选工具正文会
等到完整解析后再发送，防止客户端收到半截工具 JSON。

需要最终 usage 时设置：

```json
{"stream_options": {"include_usage": true}}
```

精确 usage 位于结束前的独立 chunk，不会阻塞首个正文或 reasoning 增量。

## 非流式响应

`stream: false` 或省略 `stream` 时，代理收齐上游流后返回一个标准 completion 对象。
推理文本位于 `choices[0].message.reasoning_content`，工具调用位于
`choices[0].message.tool_calls`。

## 错误

请求校验和上游错误使用 OpenAI 风格的 `error` 对象。流已经开始后发生的错误以 SSE
错误 chunk 结束；代理不会先返回部分内容再重放整轮。
