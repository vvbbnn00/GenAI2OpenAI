# OpenAI Responses

接口：

- `POST /v1/responses`
- `POST /v1/responses/input_tokens`

## 输入

`input` 可以是字符串，也可以是 Responses item 数组：

```json
{
  "model": "qwen-instruct",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "描述这张图"},
        {"type": "input_image", "image_url": "https://example.test/image.png"}
      ]
    }
  ],
  "stream": true
}
```

代理支持常用的 `message`、`function_call`、`function_call_output` 和 `reasoning`
item，并转换到统一消息表示。工具结果回合默认仍允许继续调用工具；只有调用方明确
给出 `tool_choice: "none"` 时才关闭。

Responses 风格的指定函数选择为：

```json
{"tool_choice": {"type": "function", "name": "get_weather"}}
```

## 流事件

流以 `response.created` 开始。正文、推理和工具调用分别使用结构化事件，包括：

- `response.output_item.added`
- `response.content_part.added`
- `response.reasoning_text.delta`
- `response.output_text.delta`
- `response.function_call_arguments.delta`
- 对应的 `done` 事件
- `response.completed` 或 `response.failed`

每个事件使用稳定的 `item_id`、`output_index`、`content_index` 和递增
`sequence_number`。reasoning item 与相邻工具调用保持独立，不会被拼入可见正文。

`response.created` 在精确 token 计算前发送，避免 tokenizer 下载或 usage 计算拖慢
流的建立。完整 usage 在 `response.completed` 中给出。

## 非流式响应

非流式实现复用同一事件生成器并在服务端收集结果，因此流式和非流式的消息转换、
工具解析和 usage 规则一致。返回对象包含 `output` 和便于读取的 `output_text`。

## 输入 token 计数

把与 `/v1/responses` 相同的请求体发送到 `/v1/responses/input_tokens`：

```json
{
  "model": "deepseek-pro",
  "input": "你好",
  "tools": []
}
```

响应：

```json
{"object": "response.input_tokens", "input_tokens": 8}
```

数值示例只说明响应形状，具体 token 数由模型、消息和当前固定 codec 决定。计数会
经过与实际生成相同的模型解析、reasoning 归一和工具 prompt 组装。
