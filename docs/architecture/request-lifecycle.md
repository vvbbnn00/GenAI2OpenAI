# 请求生命周期

## 公共路径

每个聊天请求依次经过以下阶段：

1. API 层校验 JSON、认证和协议字段。
2. 消息被转换为统一的文本、图片、tool call 和 tool result 表示。
3. 模型目录解析用户给出的 ID，并根据实际目录记录选择 adapter。
4. reasoning effort 按模型能力归一化。
5. adapter 用官方模板或模型专用规则准备 prompt 和工具声明。
6. 同一份最终消息用于 token 计数和 GenAI 请求组装。
7. 上游 SSE 被增量解析为 reasoning、正文、工具调用、usage 或错误。
8. API 层把内部事件转换为 Chat Completions、Responses 或 Anthropic 事件。

模型 ID 不能只靠字符串猜测。目录记录中的 `aiType`、`aiName`、`simpleName`、
`descInfo`、`rootAiType` 和 `rootModelName` 会共同决定实际模型和 transport。非
Xinference 记录不会套用 Xinference 专用 prompt。

## 无工具请求

没有工具时，reasoning 和正文增量在解析完成后立即转发。代理不会等待整轮生成，
也不会先做结束时才需要的精确 completion token 计算。流结束后再发送 usage。

## 工具请求

工具语法可能跨越多个上游 SSE chunk。为了避免向客户端泄露半截 XML、DSML 或
JSON，正文和候选工具调用会先缓冲到本轮完整结束，再解析和校验。独立的 reasoning
增量通常仍可实时转发。

解析成功后，代理返回客户端协议中的工具调用。客户端执行工具并把结果送回后，下一
轮仍携带工具定义，直到任务完成或调用方明确设置 `tool_choice: "none"`。

Kimi K3 对畸形桥接输出最多进行三轮有界修复。`tool_choice: "required"` 和指定函数
同样走有界循环，不会无限重试。

Kimi 续轮若带有最近已完成动作，还会比较候选 action 的工具名和规范化参数。同名
同参的候选 action 会先被暂扣并要求模型重新确认；最后一次仍重复时放行，以兼容
确实需要轮询或幂等重试的任务。这类续轮会按尝试缓冲 reasoning，只转发最终接受
尝试的内容，避免客户端看到已被丢弃的旧计划。其他工具请求保持实时 reasoning 路径。

## 重试边界

临时网络错误、可重试 HTTP 状态或无效上游响应可以重试，但仅限客户端还没有收到
任何增量时。一旦 reasoning 或正文已经交付，代理不会从头重放同一轮，以免客户端
收到重复或互相冲突的内容。

认证失败最多触发一次 token 刷新。上游流在 90 秒内没有任何字节时视为停滞；停滞
重试最多一次，并同样受“尚未向客户端发送数据”的约束。

## Kimi K3 分支

GenAI 网页传输要求 Kimi K3 当前轮用户输入位于非空 `chatInfo`。代理把最后一条用户
内容移到该字段，历史消息仍放在 `messages`。空文本或仅图片轮使用不可见占位符保持
`chatInfo` 非空。

这是上海科技大学 GenAI 网页通道的行为，不是 Moonshot 模型格式。聊天请求始终不
包含 `chatGroupId`。服务端生成的历史记录只在成功完成后通过独立清理流程定位和
删除，详情见 [Kimi K3](../models/kimi-k3.md)。
