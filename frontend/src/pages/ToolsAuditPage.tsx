import { Typography } from "antd";

export default function ToolsAuditPage() {
  return (
    <div style={{ padding: 24 }}>
      <Typography.Title level={3}>工具审计</Typography.Title>
      <Typography.Paragraph type="secondary">
        工具调用记录、审计日志 — 由后续 Issue 实现
      </Typography.Paragraph>
    </div>
  );
}
