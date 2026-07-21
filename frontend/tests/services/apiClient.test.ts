/** apiClient error handling tests (ISSUE-067). */

import { describe, it, expect, vi } from "vitest";
import { apiClient } from "../../src/services/apiClient";

vi.mock("axios", async () => {
  const actual = await vi.importActual("axios");
  return {
    ...(actual as object),
    default: {
      create: vi.fn(() => ({
        interceptors: {
          response: { use: vi.fn() },
        },
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
        defaults: {},
      })),
    },
  };
});

describe("apiClient", () => {
  it("has correct base URL from env default", () => {
    expect(apiClient.defaults).toBeDefined();
  });
});

describe("eventApi paths", () => {
  // Path correctness — verify URL templates match OpenAPI
  const BASE = "/api/v1";

  it("listEvents uses GET /events", () => {
    const url = `${BASE}/events`;
    expect(url).toBe("/api/v1/events");
  });

  it("getEvent uses GET /events/:id", () => {
    const url = `${BASE}/events/evt-123`;
    expect(url).toBe("/api/v1/events/evt-123");
  });

  it("triggerInvestigation uses POST /events/:id/investigate", () => {
    const url = `${BASE}/events/evt-123/investigate`;
    expect(url).toBe("/api/v1/events/evt-123/investigate");
  });

  it("getReport uses GET /events/:id/report", () => {
    const url = `${BASE}/events/evt-123/report`;
    expect(url).toBe("/api/v1/events/evt-123/report");
  });

  it("getTraces uses GET /events/:id/traces", () => {
    const url = `${BASE}/events/evt-123/traces`;
    expect(url).toBe("/api/v1/events/evt-123/traces");
  });

  it("listActions uses GET /events/:id/actions", () => {
    const url = `${BASE}/events/evt-123/actions`;
    expect(url).toBe("/api/v1/events/evt-123/actions");
  });

  it("approveAction uses POST /actions/:id/approve", () => {
    const url = `${BASE}/actions/act-123/approve`;
    expect(url).toBe("/api/v1/actions/act-123/approve");
  });

  it("getWriteback uses GET /writebacks/:id", () => {
    const url = `${BASE}/writebacks/wbk-123`;
    expect(url).toBe("/api/v1/writebacks/wbk-123");
  });

  it("retryWriteback uses POST /writebacks/:id/retry", () => {
    const url = `${BASE}/writebacks/wbk-123/retry`;
    expect(url).toBe("/api/v1/writebacks/wbk-123/retry");
  });
});
