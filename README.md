# NIM Claude Proxy & Admin Dashboard

专属于 **NVIDIA NIM (`nvidia_nim`)** 的轻量级高可用转接代理服务，自带全功能可视化 **Admin Dashboard 管理控制台**。

支持将 **Claude Code**、**Codex CLI**、**Pi** 等 AI 编程工具透明接入到 NVIDIA NIM 微服务节点上，并实现多 API Key 的平滑轮询（Round-Robin）、熔断冷却自动切流（Failover Cooldown）及模型别名路由。

---

## 核心特性

- **多 Key 轮询与熔断管理**：支持配置多个 `NVIDIA_NIM_API_KEYS`，自动负载均衡；当某个 Key 遇到 `429 Rate Limit` 或 `5xx` 时自动冷却隔离（例如 60s）并自动尝试下一个 Key；当遇 `401` 时自动剔除。
- **现代化可视化 Admin 控制台**：访问 `http://localhost:8000/admin` 即可进入高颜值的暗色控制台，实时监控 Key 状态与在线配置。
- **丰富的配置吸收与管理**：
  - **模型别名路由**：`NIM_DEFAULT_MODEL`、`MODEL_SONNET`、`MODEL_OPUS`、`MODEL_HAIKU`、`MODEL_FABLE`。
  - **推理思考调优**：`REASONING_PREFERENCE`、`DEFAULT_REASONING_BUDGET`。
  - **服务与安全**：`PORT`、`HOST`、`PROXY_API_TOKEN` (鉴权 Bearer Token)、`HTTP_TIMEOUT`、`NVIDIA_NIM_PROXY`。
- **Anthropic Messages & OpenAI Responses 协议双向转换**：暴露 `/v1/messages` (Claude Code) 与 `/v1/responses` (Codex)。`/v1/responses` 会将 OpenAI Responses 请求转换为 Chat Completions 格式发送给 NIM，再把结果转回 Responses 形态返回。
- **本地网络工具透明代理**：当上游模型调用 `web_search` / `web_fetch` 时，代理在本地执行（DuckDuckGo / Tavily / 豆包 / 网页抓取）并把结果回填对话，客户端无需感知；非网络类工具调用仍原样交还客户端。流式请求含网络工具时改为服务端缓冲解析后回放最终文本。
- **MiniMax / Nemotron 原生工具流清洗**：实时过滤和清洗上游暴露的原生 XML 标记（`]<]minimax[>...`），转换为合规的 Tool Call Deltas。

---

## 快速开始

```bash
# 1. 复制配置文件
cp .env.example .env

# 2. 启动服务
uv run nim-proxy
```

启动后：
- **代理服务地址**：`http://localhost:8000`
- **可视化 Admin 控制台**：`http://localhost:8000/admin`

> ⚠️ **安全提示**：一旦配置了 `PROXY_API_TOKEN`，所有 `/admin/api/*` 管理接口都必须携带该 Token（`Authorization: Bearer <token>`），否则返回 401；静态控制台页面 `/admin` 本身仍可匿名访问，但加载数据时需要填入 Token。生产环境请务必设置强 Token，避免 API Key 与 `.env` 配置被未授权读取或篡改。

---

## 连接 Claude Code

在终端中设置代理环境变量并启动 Claude Code：

```bash
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_API_KEY="dummy-key" # 代理内部会自动调度真实的 NIM Key
claude
```
