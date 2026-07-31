# ADR 0003：Kimi K3 的 GenAI 网页传输

- 状态：已采用
- 范围：通过上海科技大学 GenAI `/htk/chat/start/chat` 调用 Kimi K3

## 背景

当前 GenAI 网页通道对 Kimi K3 有两个已观察约束：

1. 当前轮用户输入必须位于非空 `chatInfo`。
2. 原生 message-level 工具声明、结构化工具历史和历史 reasoning 不会忠实透传。

非空 `chatInfo` 还会生成历史记录。请求不能预设 `chatGroupId`，该 ID 由服务端创建。

这些是 GenAI transport 行为，不是 Moonshot 官方 Kimi prompt 规则。

## 决定

普通消息、视觉和 token 继续使用固定官方 Kimi K3 codec。传输时把当前 user 内容移到
非空 `chatInfo`，其余历史放在 `messages`，且最终聊天 payload 禁止出现
`chatGroupId`。

工具请求使用项目自定义的 function-only 响应桥接：`<k3_action>` 表示调用，
`<k3_final>` 表示完成。桥接通过普通 system 消息发送，完整响应经过 JSON、函数名和
结构验证后才转换为客户端工具事件。畸形输出最多修复三轮。

工具结果压成普通 user 结果记录，最近一轮 reasoning 只保留一份 continuation
checkpoint，避免丢失计划后重复已经完成的调用。

请求前记录同问题的历史 group ID 快照，成功后只删除唯一的新匹配记录。找不到唯一
目标时跳过，清理失败不改变模型响应。

## 结果

- 现有 GenAI 通道下可以提供有界、可校验的 function tool calling。
- 该桥接不能称为 Moonshot 官方 tool calling，也不能假设具备原生协议的全部能力。
- 工具正文需要完整缓冲，reasoning 仍可实时转发。
- 长任务应把工具设计为幂等，并把桥接失败视为可恢复的外部服务故障。
- 如果未来 GenAI 原生透传 Kimi 工具声明，应优先删除桥接，切换到官方 XTML 路径。
