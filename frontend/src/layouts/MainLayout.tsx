/** Main layout — Ant Design sidebar + header + content (ISSUE-067). */

import { useState } from "react";
import { Outlet, useNavigate, useLocation } from "react-router-dom";
import { Layout, Menu } from "antd";
import type { MenuProps } from "antd";
import {
  UnorderedListOutlined,
  CheckCircleOutlined,
  ToolOutlined,
} from "@ant-design/icons";

const { Header, Sider, Content } = Layout;

const menuItems: MenuProps["items"] = [
  { key: "/events", icon: <UnorderedListOutlined />, label: "事件看板" },
  { key: "/approvals", icon: <CheckCircleOutlined />, label: "审批中心" },
  { key: "/tools-audit", icon: <ToolOutlined />, label: "工具审计" },
];

export default function MainLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  // Determine selected key from current path
  const selectedKey = location.pathname.startsWith("/events")
    ? "/events"
    : location.pathname.startsWith("/approvals")
      ? "/approvals"
      : location.pathname.startsWith("/tools-audit")
        ? "/tools-audit"
        : "/events";

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Sider collapsible collapsed={collapsed} onCollapse={setCollapsed}>
        <div
          style={{
            height: 48,
            margin: 16,
            color: "#fff",
            fontWeight: 600,
            fontSize: collapsed ? 14 : 18,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            whiteSpace: "nowrap",
            overflow: "hidden",
          }}
        >
          {collapsed ? "ST" : "ShadowTrace"}
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[selectedKey]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>
      <Layout>
        <Header
          style={{
            background: "#fff",
            padding: "0 24px",
            borderBottom: "1px solid #f0f0f0",
            display: "flex",
            alignItems: "center",
            fontSize: 16,
            fontWeight: 500,
          }}
        >
          ShadowTrace — 多 Agent 安全运营系统
        </Header>
        <Content style={{ margin: 16 }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}
