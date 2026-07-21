/** Socket.IO event types — matching contracts/socketio/events.schema.json */

import type { EventStatus, WritebackStatus } from "./event";

export interface SocketStateChange {
  event_id: string;
  old_status: EventStatus;
  new_status: EventStatus;
  timestamp: string;
}

export interface SocketEventCreated {
  event_id: string;
  event_type: string;
  title: string;
  severity: string;
  timestamp: string;
}

export interface SocketWritebackUpdated {
  event_id: string;
  writeback_id: string;
  status: WritebackStatus;
  timestamp: string;
}

/** Discriminated union of all socket events */
export type SocketEvent =
  | { type: "event_created"; payload: SocketEventCreated }
  | { type: "state_change"; payload: SocketStateChange }
  | { type: "writeback_updated"; payload: SocketWritebackUpdated };
