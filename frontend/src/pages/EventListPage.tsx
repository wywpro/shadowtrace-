import { Typography } from "antd";

export default function EventListPage() {
  return (
    <div style={{ padding: 24 }}>
      <Typography.Title level={3}>事件看板</Typography.Title>
      <Typography.Paragraph type="secondary">
        事件列表、状态筛选、分页 — 由 ISSUE-068 实现
      </Typography.Paragraph>
    </div>
  );
}
