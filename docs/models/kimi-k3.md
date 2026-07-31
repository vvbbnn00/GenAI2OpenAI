# Kimi K3

Kimi K3 同时涉及官方模型 codec 和上海科技大学 GenAI 网页传输。两者必须分开描述。

## 官方 codec

- 仓库：[moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
- 固定 revision：`9f62e4e9fffbd0a83ddd60e1c209d828994b3569`
- `tiktoken.model` SHA-256：
  `b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103`
- `encoding_k3.py` SHA-256：
  `b9cb7ae100fed34b9337f80dacee5abbf7e261fe9b74bc0e76366701d46f5333`

普通 system/user/assistant 消息、文本 token、官方 XTML 返回解析和视觉 token 使用
这些固定资源。显式 OpenAI effort 会归一为 max，因为 GenAI 通道没有独立 K3
effort 字段；这不会额外伪造一个上游 `thinking` 布尔字段。

## 视觉输入

Kimi 支持 URL 和 base64 data URL 图片。视觉 token 规则来自固定 revision：

- patch size：14
- merge kernel：2
- 单边 patch 上限：512
- 输入 patch 总量上限：65,536

图片被映射到 GenAI 的 `imageUrl`、`imageUrls`、`width` 和 `height` 字段。视觉尺寸
参与官方 token 计算。

## `chatInfo` 传输约束

通过 GenAI `/htk/chat/start/chat` 调用 Kimi 时，当前轮用户输入必须位于非空
`chatInfo`。否则上游会形成缺少有效 `content` 的用户消息并拒绝请求。代理因此：

1. 把最后一条 user 内容移到 `chatInfo`。
2. 把此前历史保留在 `messages`。
3. 对空文本、仅图片或工具续接轮使用不可见占位符，使 `chatInfo` 非空。

这个约束来自 GenAI 网页服务，不是 Kimi K3 官方 prompt 格式。其他模型可以把完整
用户输入放在 `messages` 并保持 `chatInfo` 为空。

## 明确禁止 `chatGroupId`

代理的聊天请求 payload 不包含 `chatGroupId`。该 ID 由 GenAI 服务端生成，不能由
客户端预设或复用。测试会直接检查最终 payload，防止以后重构时误加该字段。

## 历史记录清理

非空 `chatInfo` 会在 GenAI 历史接口留下记录。Kimi 请求开始前，代理按本轮问题内容
取得现有 group ID 快照；成功结束后轮询新增记录，只在恰好找到一条新匹配记录时
调用删除接口。

相同问题的并发请求由内容 hash 对应的锁串行定位，降低交叉删除风险。找不到记录、
找到多条、生成失败或历史接口异常时都跳过删除。清理失败只写日志，不改变已生成的
模型响应。

## Tool calling 桥接

GenAI 网页通道不会透传 Kimi 原生 message-level `tools`、`tool_choice`、结构化工具
历史或历史 `reasoning_content`。因此当前通道无法忠实发送 Moonshot 官方
`type="tool-declare"` XTML。

项目只在工具请求中加入一个普通 system 桥接消息，定义两种响应信封：

```text
<k3_action>{"name":"tool_name","arguments":{...}}</k3_action>
<k3_final>最终答复</k3_final>
```

这不是 Moonshot 官方协议。代理会完整收齐响应，再校验信封、JSON、工具名和参数
结构。普通 JSON 在 auto 模式下不会被误执行。畸形结果和 required 未调用工具的
情况最多修复三轮。

项目不再生成 `Call-expression schemas (JSON):`、`User request:` 或
`name(key=value)` 正文。若模型直接返回官方 XTML，原生解析器仍能识别，便于未来
上游提供原生透传后切换。

## 长工具链续接

GenAI 会丢弃历史 `reasoning_content`，直接重放普通工具历史容易让模型把已完成动作
重新规划。代理把每个完成结果压成 `Completed client action result:` 用户消息，并
只为最近工具轮保留一份 continuation checkpoint。checkpoint 列出已完成调用和最近
推理状态，要求从第一个未完成步骤继续。

旧 reasoning 不累计，已完成动作也不以当前动作标签重放。这样控制上下文增长，并
降低长会话中重复执行同一工具的概率，但它不能把受限网页通道变成原生 tool calling
API。调用方仍应给工具设置幂等保护和合理超时。

## 流式边界

reasoning 增量可以在工具响应完成前转发。`<k3_action>`、`<k3_final>` 及其正文要等
信封闭合并验证成功后再交付，防止客户端收到半截 JSON 或协议标签。这一小段缓冲是
解析正确性的边界，不是对整轮 reasoning 的缓冲。
