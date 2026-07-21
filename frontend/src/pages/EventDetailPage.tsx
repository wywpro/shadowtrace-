import { useParams } from "react-router-dom";
import { Typography } from "antd";

export default function EventDetailPage() {
  const { eventId } = useParams<{ eventId: string }>();
  return (
    <div style={{ padding: 24 }}>
      <Typography.Title level={3}>事件详情</Typography.Title>
      <Typography.Paragraph type="secondary">
        event_id: {eventId} — 由 ISSUE-069 实现
      </Typography.Paragraph>
    </div>
  );
}
