import { Typography } from "antd";

export default function ApprovalPage() {
  return (
    <div style={{ padding: 24 }}>
      <Typography.Title level={3}>审批中心</Typography.Title>
      <Typography.Paragraph type="secondary">
        审批队列、历史记录 — 由后续 Issue 实现
      </Typography.Paragraph>
    </div>
  );
}
