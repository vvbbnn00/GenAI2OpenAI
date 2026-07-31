# 模型适配总览

模型适配不是只按请求中的字符串选择。代理先从 GenAI 模型目录取得实际记录，再综合
`aiType`、名称和描述字段判断模型家族。这样即使公开 ID 没有写明版本，仍可区分
DeepSeek V4 Flash/Pro、GLM 5.2 和旧模型。

## 当前维护范围

| GenAI ID | Adapter | Prompt/codec | 视觉 | 上游 thinking 开关 |
| --- | --- | --- | --- | --- |
| `chatglm` | `glm_5_2` | GLM 5.2 官方模板 | 否 | 无独立字段 |
| `deepseek-chat` | `deepseek_v4_flash` | DeepSeek V4 Flash 官方 encoder | 否 | 支持 |
| `deepseek-pro` | `deepseek_v4_pro` | DeepSeek V4 Pro 官方 encoder | 否 | 支持 |
| `qwen-instruct` | `qwen_3_5` | Qwen 3.5 官方模板 | 是 | 无独立字段 |
| `kimi-k3` | `kimi_k3` | Kimi K3 官方 codec，工具使用通道桥接 | 是 | 无独立字段 |

模型目录记录明确显示 `rootModelName` 不是 Xinference 时，代理不会应用这些 adapter。
没有匹配官方公开 tokenizer 的模型仍可走通用代理路径，但 token 数是兼容估算。

## 共同约束

- 官方资源固定到完整 commit revision，并在使用前校验 SHA-256。
- 工具 prompt、token 计数和 completion 序列化使用同一个模型家族 codec。
- 图片只允许出现在 user 消息中，目前只交给 Qwen 3.5 或 Kimi K3。
- reasoning 增量会尽可能实时转发；候选工具语法在完整验证前不会暴露给客户端。
- 所有 GenAI 聊天请求都不发送 `chatGroupId`。

各模型的具体差异见：

- [GLM 5.2](glm-5.2.md)
- [DeepSeek V4 Flash 与 Pro](deepseek-v4.md)
- [Qwen 3.5](qwen-3.5.md)
- [Kimi K3](kimi-k3.md)

`src/genai_proxy/models/legacy/` 中的 MiniMax 等代码只保留历史兼容，不进入当前故障
回退目录或在线模型矩阵。
