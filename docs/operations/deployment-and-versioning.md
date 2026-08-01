# 部署与版本信息

## 直接运行

开发和单机调试可使用：

```bash
uv run main.py --keystore /path/to/ids-passkey.keystore
```

也可以使用 `uv run python -m genai_proxy` 或 `uv run genai2openai`。三者进入同一个
CLI。开发服务器使用 Flask threaded 模式，生产环境应使用 WSGI 入口。

## Docker Compose

```bash
cp .env.example .env
./scripts/docker-compose.sh up -d --build
```

镜像安装 wheel 包，并由 Gunicorn 加载 `genai_proxy.wsgi:app`。Dockerfile 只复制
`pyproject.toml`、`README.md` 和 `src/`。`.dockerignore` 同时排除 `.git`、docs、
tests、smoke 脚本、环境文件和 keystore。

模型目录缓存使用独立 volume。官方 Hugging Face 资源在镜像构建阶段下载、校验并
加载，之后保存在镜像内的只读目录；容器运行时强制离线读取，不使用 tokenizer volume。
keystore 使用 bind mount，且必须可写，因为认证器会更新 passkey 计数器。

## 版本日志

每次启动都会输出以下三种形式之一：

```text
Program version: commit=<full-hash> committed_at=<ISO-8601> source=git
Program version: commit=<full-hash> committed_at=<ISO-8601> source=image
Program version: local-dev
```

如果参与运行或镜像构建的源码有未提交改动，还会附加 `dirty=true`。

直接运行时，版本模块在仓库根目录读取 Git。读取不到合法 hash 和带时区提交时间时
回退到 `local-dev`。

Docker 不复制 `.git`。`scripts/docker-compose.sh` 在宿主机读取最后一次 commit，把
hash、提交时间和脏状态作为短构建参数交给 Dockerfile；镜像把结果写入
`genai_proxy/_build_version.json`。运行容器不需要 Git 仓库。

源码归档或 CI 没有 `.git` 时，可以显式设置构建参数。只有当参数确实对应待构建源码
时才这样做，否则部署日志会给出错误的版本归属。

## 多机器同步

建议以启动日志中的完整 hash 为部署标识：

1. 在构建机确认工作区干净并记录完整 hash。
2. 使用包装脚本构建镜像。
3. 给镜像添加包含该 hash 的不可变 tag。
4. 每台机器启动后核对 `source=image`、完整 hash 和提交时间。
5. 任一实例出现 `dirty=true` 或 `local-dev` 时，不把它视为同一可复现版本。

## 反向代理

应用为 SSE 响应设置 `Cache-Control: no-cache, no-transform` 和
`X-Accel-Buffering: no`。外层 Nginx、CDN 或负载均衡器仍需关闭响应缓冲，并把读取
超时设置到覆盖最长模型请求。Gunicorn 默认超时是 180 秒；外层超时不应更短。
