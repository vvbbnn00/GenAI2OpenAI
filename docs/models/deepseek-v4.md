# DeepSeek V4 Flash 与 Pro

## 型号与固定资源

| GenAI ID | 官方仓库 | 固定 revision |
| --- | --- | --- |
| `deepseek-chat` | [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | `60d8d70770c6776ff598c94bb586a859a38244f1` |
| `deepseek-pro` | [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | `b5968e9190ef611bbf34a7229255be88a0e937c1` |

两者在当前固定 revision 使用相同的：

- `tokenizer.json` SHA-256：
  `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`
- `encoding/encoding_dsv4.py` SHA-256：
  `bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0`

Flash 和 Pro 使用不同 adapter，避免后续官方资源变化时错误共用版本。

## 第一方 prompt 传输

GenAI 不会把顶层结构化 `tools` 交给 DeepSeek V4。代理没有改用手写 schema，而是
调用固定 revision 的官方 `encoding_dsv4.py`，让它渲染完整多轮 prompt、reasoning
前缀、DSML 工具声明、历史调用和工具结果。

随后，代理只把这段官方 prompt 无损拆进 GenAI 接受的 system/user 消息外壳。组装
后会重新经过官方 encoder，并要求结果与原 prompt 逐字节相等。边界变化会直接报错，
不会静默发送一个近似模板。

## Thinking 和 effort

DeepSeek V4 是当前活动模型中唯一使用 GenAI 顶层 `thinking` 布尔字段的家族：

| 客户端配置 | 规范 effort | 上游 `thinking` |
| --- | --- | --- |
| 未提供 reasoning | 无 | `false` |
| `none` | `none` | `false` |
| `minimal`、`low`、`medium`、`high` | `high` | `true` |
| `xhigh`、`max` | `max` | `true` |

只有 max 档会加入官方 encoder 定义的 `REASONING_EFFORT_MAX` 前缀。high 使用官方
thinking 模式但不加入 max 前缀。

Anthropic `thinking.enabled` 或 `adaptive` 未指定 effort 时映射到 high；
`thinking.disabled` 映射到 none。

## Tool calling

工具声明和结果使用官方 DSML `<｜DSML｜tool_calls>` 结构，并兼容旧的
`<｜DSML｜function_calls>` 返回。代理会验证函数名和参数 JSON，再转换成客户端协议。

`tool_choice` 中无法由官方 encoder 表达的调用方限制会作为普通 system 内容加入，
然后仍由同一 encoder 处理。

## 流式输出

Flash 和 Pro 的 reasoning 都从上游 SSE 增量转发。工具请求的 reasoning 不等待完整
工具调用；DSML 正文会等闭合和解析成功后再交付。当前不支持视觉输入。
