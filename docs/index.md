# GenAI2OpenAI Wiki

这里记录项目当前的接口契约、模型适配和运维方式。代码仍是最终事实来源；文档中的
固定 revision、默认值和测试命令都对应当前仓库实现。

## 开始使用

- [安装与首次请求](getting-started.md)
- [配置参考](operations/configuration.md)
- [部署与版本信息](operations/deployment-and-versioning.md)

## 了解实现

- [架构总览](architecture/overview.md)
- [请求生命周期](architecture/request-lifecycle.md)
- [消息、prompt 与 token](architecture/messages-prompts-tokens.md)
- [可靠性与失败边界](operations/reliability.md)

## API

- [OpenAI Chat Completions](api/chat-completions.md)
- [OpenAI Responses](api/responses.md)
- [Anthropic Messages](api/anthropic-messages.md)

## 模型

- [模型适配总览](models/index.md)
- [GLM 5.2](models/glm-5.2.md)
- [DeepSeek V4 Flash 与 Pro](models/deepseek-v4.md)
- [Qwen 3.5](models/qwen-3.5.md)
- [Kimi K3](models/kimi-k3.md)

## 开发

- [测试与验证](development/testing.md)
- [增加或更新模型适配](development/adding-a-model.md)

## 架构决策

- [ADR 0001：固定官方 Hugging Face codec](decisions/0001-official-hf-codecs.md)
- [ADR 0002：统一规范消息](decisions/0002-canonical-messages.md)
- [ADR 0003：Kimi K3 的 GenAI 网页传输](decisions/0003-kimi-web-transport.md)

## 支持范围

重点维护和在线验证的模型是 GLM 5.2、DeepSeek V4 Flash、DeepSeek V4 Pro、
Qwen 3.5 和 Kimi K3。`src/genai_proxy/models/legacy/` 中保留的适配仅用于兼容旧
导入或历史测试，不代表当前 GenAI 上游仍提供对应模型。
