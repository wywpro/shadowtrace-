/** Event store — zustand slice for event list + cache (ISSUE-067). */

import { create } from "zustand";
import type { EventListItem, EventDetailResponse, EventListParams } from "../types/event";
import { socketClient } from "../services/socketClient";
import { listEvents } from "../services/eventApi";

interface EventState {
  // Event list
  items: EventListItem[];
  total: number;
  loading: boolean;
  error: string | null;

  // Current event detail cache
  currentEvent: EventDetailResponse | null;

  // Polling
  pollInterval: number; // ms, 0 = disabled
  pollTimer: ReturnType<typeof setInterval> | null;
  socketUnsub: (() => void) | null;

  // Actions
  loadEvents: (params?: EventListParams) => Promise<void>;
  setCurrentEvent: (event: EventDetailResponse | null) => void;
  startPolling: (intervalMs?: number) => void;
  stopPolling: () => void;

  // Internal socket-driven updates
  _applySocketUpdate: (eventId: string, patch: Partial<EventListItem>) => void;
  _insertEvent: (item: EventListItem) => void;
}

export const useEventStore = create<EventState>((set, get) => ({
  items: [],
  total: 0,
  loading: false,
  error: null,
  currentEvent: null,
  pollInterval: 10_000,
  pollTimer: null,
  socketUnsub: null,

  async loadEvents(params) {
    set({ loading: true, error: null });
    try {
      const res = await listEvents(params);
      set({ items: res.data.items, total: res.data.total, loading: false });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to load events";
      set({ error: message, loading: false });
    }
  },

  setCurrentEvent(event) {
    set({ currentEvent: event });
  },

  startPolling(intervalMs) {
    const { pollTimer } = get();
    if (pollTimer) return; // already polling
    const ms = intervalMs ?? get().pollInterval;
    if (ms <= 0) return;
    const timer = setInterval(() => {
      get().loadEvents();
    }, ms);
    set({ pollTimer: timer });

    // Also connect socket for real-time updates; poll is fallback.
    // Save the unsubscriber so stopPolling can clean up (Should-Fix #1).
    socketClient.connect();
    const unsub = socketClient.onEvent((evt) => {
      if (evt.type === "event_created") {
        // will appear on next poll
        get().loadEvents();
      } else if (evt.type === "state_change") {
        get()._applySocketUpdate(evt.payload.event_id, {
          status: evt.payload.new_status,
        });
      } else if (evt.type === "writeback_updated") {
        get()._applySocketUpdate(evt.payload.event_id, {
          writeback_overall_status: evt.payload.status,
        });
      }
    });
    set({ socketUnsub: unsub });
  },

  stopPolling() {
    const { pollTimer, socketUnsub } = get();
    if (pollTimer) {
      clearInterval(pollTimer);
      set({ pollTimer: null });
    }
    socketUnsub?.();
    socketClient.disconnect();
    set({ socketUnsub: null });
  },

  _applySocketUpdate(eventId, patch) {
    set((s) => ({
      items: s.items.map((item) =>
        item.event_id === eventId ? { ...item, ...patch } : item,
      ),
    }));
  },

  _insertEvent(item) {
    set((s) => ({ items: [item, ...s.items], total: s.total + 1 }));
  },
}));
