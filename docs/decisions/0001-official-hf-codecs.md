# ADR 0001：固定官方 Hugging Face codec

- 状态：已采用
- 范围：GLM 5.2、DeepSeek V4、Qwen 3.5、Kimi K3

## 背景

手工复制模型 prompt 容易漏掉特殊 token、reasoning 边界、工具历史和视觉占位。仓库
`main` 分支也会变化，同一项目版本可能在不同机器得到不同 prompt 或 token 数。

## 决定

每个活动模型固定到官方 Hugging Face 的完整 commit revision，并记录所需文件的
SHA-256。运行时按 revision 下载资源，hash 校验成功后原子写入缓存。

工具 prompt 优先直接调用官方 `chat_template.jinja` 或 Python encoder。输入 token、
completion token 和视觉 token 使用同一个模型家族 codec。若模板边界与适配器预期
不一致，直接失败，不退回近似手写模板。

DeepSeek 的官方 Python encoder 会在 hash 校验后加载并执行。固定 revision 和
SHA-256 把执行内容锁定到代码审查过的字节，避免仓库 `main` 后续变化直接进入运行
环境；更新 encoder 时必须像更新本项目源码一样审查完整差异。Jinja 模板则在不可变
sandbox 环境中渲染。

## 结果

- 同一 commit 在多机器产生可复现的 prompt 和 token 数。
- 官方模板升级需要显式更新 revision、hash 和差分测试。
- 第一次使用模型需要联网下载资源，生产部署应持久化 tokenizer 缓存。
- 执行固定的官方 Python encoder 仍是明确的供应链信任边界，hash 只能固定内容，不能
  证明内容本身安全。
- 没有官方公开 tokenizer 的模型只能提供明确标注的兼容估算。

Kimi 工具桥接不属于官方 codec。它是独立的传输决策，见
[ADR 0003](0003-kimi-web-transport.md)。
