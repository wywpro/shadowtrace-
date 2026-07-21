/** Socket.IO client wrapper with poll fallback (ISSUE-067). */

import { io, Socket } from "socket.io-client";
import type {
  SocketEvent,
  SocketEventCreated,
  SocketStateChange,
  SocketWritebackUpdated,
} from "../types/socket";

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL ?? "http://localhost:8000";

type EventHandler = (event: SocketEvent) => void;

class SocketClient {
  private socket: Socket | null = null;
  private handlers: Set<EventHandler> = new Set();
  private connected = false;

  /** Connect to the global room. Safe to call multiple times (dedup). */
  connect(): void {
    if (this.socket?.connected) return;
    try {
      this.socket = io(SOCKET_URL, {
        transports: ["websocket", "polling"],
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionAttempts: 10,
        timeout: 5000,
      });
      this.socket.on("connect", () => {
        this.connected = true;
      });
      this.socket.on("event_created", (payload: SocketEventCreated) => {
        this.emit({ type: "event_created", payload });
      });
      this.socket.on("state_change", (payload: SocketStateChange) => {
        this.emit({ type: "state_change", payload });
      });
      this.socket.on("writeback_updated", (payload: SocketWritebackUpdated) => {
        this.emit({ type: "writeback_updated", payload });
      });
      this.socket.on("disconnect", () => {
        this.connected = false;
      });
    } catch {
      // Socket unavailable — caller falls back to polling (降级策略)
      this.connected = false;
    }
  }

  disconnect(): void {
    this.socket?.disconnect();
    this.socket = null;
    this.connected = false;
  }

  get isConnected(): boolean {
    return this.connected;
  }

  onEvent(handler: EventHandler): () => void {
    this.handlers.add(handler);
    return () => {
      this.handlers.delete(handler);
    };
  }

  private emit(event: SocketEvent): void {
    for (const h of this.handlers) {
      try {
        h(event);
      } catch {
        // best-effort delivery
      }
    }
  }
}

export const socketClient = new SocketClient();
