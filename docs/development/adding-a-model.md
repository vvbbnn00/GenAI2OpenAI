# 增加或更新模型适配

模型适配应以可复现的官方资料为起点，不从模型输出反推一个近似模板。

## 1. 确认上游身份

先取得 GenAI 模型目录记录，保存 `aiType`、名称、描述、`rootAiType` 和
`rootModelName`。只有实际 Xinference 模型才进入本项目的 Xinference adapter。

如果公开 ID 会复用，识别逻辑必须结合目录记录，并为同厂不同版本使用不同 adapter。

## 2. 固定官方资源

在对应的 `src/genai_proxy/models/<family>/codec.py` 中记录：

- 官方 Hugging Face 仓库
- 完整 commit revision
- 必需的 tokenizer、模板或 Python encoder 路径
- 每个文件的 SHA-256

下载必须经过 `hf_assets.py` 的原子缓存和 hash 校验。不要跟随 `main`，也不要接受
请求方指定的仓库或 revision。Python encoder 会在校验后执行，因此更新它时必须审查
完整源码，并把固定文件视为与项目源码相同的供应链信任边界。

## 3. 定义消息和 prompt

先写官方模板差分测试，再写 adapter。测试至少覆盖：

- system/developer/user/assistant 多轮消息
- reasoning 开启和关闭
- tool schema
- assistant tool call 和 tool result 历史
- generation prompt 边界
- 重新编码后与官方结果一致

模型模板支持结构化 tools 时，直接调用模板。若 GenAI transport 丢字段，应先证明
哪些字段丢失，再设计最小桥接，并在文档中明确它不是官方协议。

## 4. Token 与视觉

输入计数必须发生在最终 prompt 组装后。completion 计数使用模型原始输出序列。视觉
模型还要覆盖宽高读取、resize、patch/merge、极端比例和多个图片。

不要因为 tokenizer 不可用就返回一个看似精确的数字。通用估算必须与官方精确路径
区分。

## 5. Tool calling

解析器应验证：

- 工具类型和函数名
- 参数是否为 JSON object
- required、none 和指定函数的约束
- 多工具调用顺序
- 跨 SSE chunk 的标签和转义
- 历史 tool call/result 是否能继续下一步
- 畸形输出的重试次数有明确上限

流式测试要证明 reasoning 可先到达，而半截工具语法不会泄露。

## 6. 接入与验证

更新 registry、模型目录回退记录和三个 API 的共用路径。至少通过：

1. codec 与模板差分测试
2. token 和视觉单元测试
3. tool parser 边界测试
4. transport payload 测试，包括禁止 `chatGroupId`
5. 三种 API 的流式和非流式集成测试
6. 显式 keystore 真实上游短测试
7. 适合长工具任务的模型再进入隔离 OMP runner

完成后更新模型页面，写清已验证事实、当前通道限制和仍未验证的部分。
