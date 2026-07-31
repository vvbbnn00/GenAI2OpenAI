# GLM 5.2

## 识别

GenAI ID 通常为 `chatglm`。当目录记录没有明确版本但名称和描述属于当前 GLM
Xinference 服务时，代理按 GLM 5.2 处理。明确的 GLM 5.1 记录仍使用单独的旧 revision。

## 官方资源

- 仓库：[zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2)
- 固定 revision：`b4734de4facf877f85769a911abafc5283eab3d9`
- `tokenizer.json` SHA-256：
  `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`
- `chat_template.jinja` SHA-256：
  `172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679`

工具声明由该 revision 的 `chat_template.jinja` 渲染。代理解析官方
`<tool_call>name<arg_key>...<arg_value>...` 结构，并保留对已观察到的合法变体的
有界兼容。

## Reasoning

官方模板支持 high 和 max，但 GenAI 网页接口没有暴露模板的 `reasoning_effort`
参数。模板会使用自己的默认 max。代理因此把所有显式 OpenAI effort 归一为 max，
且不再追加第二条 `Reasoning Effort` 指令，避免重复或冲突。

这表示 API 可以接受统一的 effort 字段，但无法通过当前上游通道真正切换 GLM 5.2
档位。文档和 usage 不应把 low/high 请求描述成实际生效的 GLM 档位。

## Tool calling

`auto`、`required`、`none` 和指定函数都支持。官方模板负责工具 schema 和格式；
调用方约束如果模板本身没有字段表达，会作为短 system 限制追加。

工具结果保留为结构化历史，再由同一官方模板编码。长任务不会把过往工具调用降级为
普通用户文本。

## 流式输出

上游 `reasoning_content`、`reasoning` 或完整 `<think>` 片段会统一转换为客户端的
reasoning 事件。没有工具时正文与 reasoning 都实时转发；有工具时只缓冲可能属于
工具语法的正文。

GLM 5.2 当前不启用视觉输入。
