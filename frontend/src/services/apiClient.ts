/** Typed Axios client with unified error handling (ISSUE-067). */

import axios, { AxiosError } from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

export interface ApiErrorPayload {
  error_code: string;
  error_message: string;
  details?: Record<string, unknown>;
}

export class ApiError extends Error {
  error_code: string;
  details?: Record<string, unknown>;

  constructor(payload: ApiErrorPayload) {
    super(payload.error_message);
    this.name = "ApiError";
    this.error_code = payload.error_code;
    this.details = payload.details;
  }
}

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000,
  headers: { "Content-Type": "application/json" },
});

// Response interceptor — normalise backend errors into ApiError
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiErrorPayload>) => {
    if (error.response?.data) {
      const { error_code, error_message, details } = error.response.data;
      throw new ApiError({
        error_code: error_code ?? "unknown_error",
        error_message: error_message ?? error.message,
        details,
      });
    }
    if (error.code === "ECONNABORTED") {
      throw new ApiError({ error_code: "timeout", error_message: "Request timed out" });
    }
    throw new ApiError({
      error_code: "network_error",
      error_message: error.message,
    });
  },
);

export default apiClient;
