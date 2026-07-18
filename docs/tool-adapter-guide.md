# ToolAdapter 与 DispositionAdapter 候选接入指南

本文描述 ShadowTrace 内部的厂商无关候选契约。当前没有真实 XDR 或安全设备的正式接口文档与脱敏网络证据，因此：

- 不包含、也不推断任何厂商 URL、鉴权头、operation code 或错误码。
- `HttpDispositionAdapter` 的能力默认全部为 `UNKNOWN`，默认禁止副作用。
- `FileStateFirewallAdapter` 只写本地 JSON，用于证明替换机制；它始终标记 `simulated=true`，不是生产防火墙。

## 边界与执行 owner

一次 Action 只能冻结一个 `ExecutionOwner`：

- `DIRECT_TOOL`：`ToolExecutor` 调用 `BaseToolAdapter`。执行完成后只允许通过 DispositionAdapter 同步一次 `EXECUTION_RESULT_RECORD`。
- `XDR_MANAGED`：通过 DispositionAdapter 提交 `ENTITY_ACTION_SUBMIT`，不得再调用设备 ToolAdapter。

`update_source_event_disposition` 仍是 disposition-only 的 `EVENT_STATUS_UPDATE`，不经过 ToolProvider。

## AdapterConfig

`AdapterConfig` 只保存配置和凭证引用：

| 字段 | 语义 |
| --- | --- |
| `endpoint` | 适配器自身解释的明确端点或本地文件路径 |
| `auth_type` | `none`、`bearer` 或 `basic` |
| `credential_ref` | 凭证所在的环境变量名，不是凭证值 |
| `timeout_s` | 单次远端调用超时 |
| `tls_verify` | 是否校验 TLS 证书；live 应保持 `true` |
| `enabled` | 是否显式启用 |

只读 Source 凭证和写回凭证必须分离。候选 HTTP Adapter 会拒绝相同的环境变量名或相同凭证值；只有取得正式文档且完成 scope 校验后，才能显式设置 `shared_credential_scope_verified=true`。

## TOOL_MODE

应用组合层通过 `configure_tool_registry` 构建独立 Registry：

- `mock`：加载 Mock 工具；要求 `simulation_enabled=true`。
- `live`：只加载调用方显式传入且启用的非模拟 Adapter，不导入 Mock 实现；`simulated=true` 的 Adapter 会被拒绝。
- `mixed`：必须提供逐工具 `mixed_routes`。未列出的工具不会自动回退；Mock 路由仍要求 `simulation_enabled=true`。

live Adapter 或 mixed 中的非模拟路由还必须显式传入
`allow_live_side_effects=true`（对应 `ALLOW_LIVE_SIDE_EFFECTS`）；默认值保持
`false`。文件示例属于 mixed 模拟路由，不可用于绕过该 live 门禁。

示例：

```python
config = AdapterConfig(
    endpoint="file:///var/lib/shadowtrace/example-firewall.json",
    auth_type="none",
    enabled=True,
)
await registry.auto_discover_for_mode(
    tool_mode="mixed",
    adapter_configs={"file_state_firewall": config},
    mixed_routes={
        "block_ip": "file_state_firewall",
        "block_domain": "mock",
    },
)
```

每个结果的 `provider_name` 由冻结后的 Registry binding 决定，不能由 Provider 响应伪造。Mock 和文件示例结果均带 `simulated=true`。健康检查失败会使工具 `available=false`，`ToolExecutor` 返回 `unsupported` 并要求人工处理，不会改走 Mock。

## BaseToolAdapter 能力

`CapabilityManifest` 必须分别声明：

- `supports_status_query`
- `supports_lookup_by_idempotency`
- `supports_idempotency`

没有声明的 `get_job_status` 或 `lookup_by_idempotency` 返回结构化 `unsupported`。异步工具沿用预创建的内部 `execution_job_id`；Provider 只能返回独立的 `provider_job_id`，不得创建或覆盖内部 job ID。

## 本地文件状态示例

`FileStateFirewallAdapter` 复用规范中的 `block_ip` 输入 Schema，并使用：

- SHA-256 形式的幂等键索引，不保存原始幂等键。
- 参数哈希检测同一幂等键的不同请求。
- 文件锁、临时文件、`fsync` 和原子替换避免半写状态。
- `lookup_by_idempotency` 与 `get_job_status` 支持进程崩溃后的查证。

该适配器完成本地写入后，异步 `block_ip` 契约先返回 `accepted`；状态查询再返回 `success`。这不是外部系统写回成功，也不能用于关闭事件。

## Generic HTTP Disposition profile

`HttpDispositionAdapter` 只向 `AdapterConfig.endpoint` 发送严格的 `DispositionCommand`，不拼接厂商路径。状态、幂等查证和健康端点均由应用组合层传入完整 URL 或模板。

默认行为：

1. 能力为 `UNKNOWN`，提交被 `writeback_unsupported` 阻断。
2. `allow_side_effects=false`，即使配置存在也不会提交。
3. 超时、传输丢失、5xx 或畸形成功响应先按幂等键查证；无法确认时返回 `UNKNOWN`，禁止盲重试。
4. 查询只发送幂等键和 Source locator 的 SHA-256，不发送原值。
5. 响应 receipt 统一脱敏。

候选测试 profile 的分类仅用于验证 ShadowTrace Adapter 边界：

| HTTP 状态 | ShadowTrace 分类 |
| --- | --- |
| 401 | `auth_error` |
| 403 | `permission_denied` |
| 409 | `version_conflict` |
| 429 | `rate_limited` |
| 5xx / 响应丢失 | 幂等查证；否则 `UNKNOWN/unknown_delivery` |

真实 Adapter 必须根据已确认协议重新建立契约测试，不能直接把这张候选表当作厂商事实。

## 人工降级

Provider 未配置、能力 `UNKNOWN`、健康检查失败或查证不确定时：

- 冻结相关 Action，不修改业务上的 `writeback_required`。
- 不回退 Mock，不生成 Mock 成功 receipt。
- `UNKNOWN` 只能继续状态/幂等查证或交由人工裁决。
