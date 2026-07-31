# ADR 0002：统一规范消息

- 状态：已采用
- 范围：Chat Completions、Responses、Anthropic Messages

## 背景

三个客户端协议使用不同的文本、图片、工具调用和工具结果结构。如果每个 adapter
直接解析外部请求，会产生三套验证和三套 token 路径，同一内容可能因 API 不同得到
不同 prompt。

## 决定

API 层先把请求转换为统一消息，再交给 chat 编排和模型 adapter。规范层负责：

- 文本与内容块顺序
- 图片 URL/data URL 校验
- assistant tool call 与 tool result 关联
- reasoning 历史
- developer/system 角色兼容

模型 adapter 只处理规范消息和模型专用模板。token 计数、生成和流式路径共享同一份
准备结果。

## 结果

- 三个 API 对非法图片、工具名和消息序列给出一致结果。
- Responses 和 Anthropic 的工具结果可继续进入同一多轮工具循环。
- 新 API 只需编写协议转换，不应复制模型 prompt 逻辑。
- 规范消息仍需保留模型需要的结构，不能为了统一而把 tool history 全部压成文本。
