/** Every backend call. Types mirror the FastAPI response models in backend/api and backend/security. */

const BASE_URL = (import.meta.env["VITE_API_BASE_URL"] as string | undefined)?.replace(/\/+$/, "") || "http://127.0.0.1:8000";
const TOKEN_KEY = "lablens_token";

export type ResultStatus = "low" | "normal" | "high" | null;
export type GenerationMode = "llm" | "template" | "insufficient" | "safe_fallback" | "unavailable";

export interface User {
  user_id: string;
  name: string;
  email: string;
}

export interface LabResult {
  test: string;
  value: number;
  unit: string | null;
  reference_range: string | null;
  confidence: number;
  needs_verification: boolean;
  status: ResultStatus;
  warning: string | null;
}

export interface SourceLink {
  title: string;
  url: string | null;
}

export interface SourceInfo {
  test_name: string;
  passages: string[];
  links: SourceLink[];
}

export interface ReportSummary {
  id: string;
  task_id: string;
  name: string;
  report_type: string | null;
  status: "approved" | "fallback";
  created_at: string;
  has_file: boolean;
  test_count: number;
  result_statuses: ResultStatus[];
  chat_count: number;
}

export interface ReportDetail extends ReportSummary {
  original_filename: string;
  content_type: string | null;
  results: LabResult[];
  final_response: string | null;
  message: string | null;
  sources: SourceInfo[];
}

export interface AnswerFinding {
  test: string;
  value: number;
  unit: string | null;
  reference_range: string | null;
  status: ResultStatus;
  what_it_measures: string;
  explanation: string;
  possible_meaning: string;
  recommended_discussion: string;
  insufficient_information: boolean;
  generation_mode: GenerationMode | null;
  sources: SourceLink[];
}

export interface ChatAnswer {
  task_id: string;
  status: "approved" | "fallback";
  findings: AnswerFinding[];
  message: string | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  created_at: string;
  text: string | null;
  test_names: string[];
  answer: ChatAnswer | null;
}

export interface ChatSummary {
  id: string;
  report_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ChatDetail extends ChatSummary {
  messages: ChatMessage[];
}

export interface SendMessageResponse {
  chat: ChatSummary;
  user_message: ChatMessage;
  assistant_message: ChatMessage;
}

export interface AuditEntry {
  log_id: string;
  task_id: string;
  report_id: string;
  user_id: string;
  agent: string;
  action: string;
  status: string;
  timestamp: string;
  details: Record<string, unknown>;
}

// --- Session token -----------------------------------------------------------

const listeners = new Set<() => void>();

export function getToken(): string | null {
  return typeof window === "undefined" ? null : window.localStorage.getItem(TOKEN_KEY);
}

function setToken(token: string | null) {
  if (typeof window === "undefined") return;
  if (token) window.localStorage.setItem(TOKEN_KEY, token);
  else window.localStorage.removeItem(TOKEN_KEY);
  listeners.forEach((fn) => fn());
}

/** Called whenever the user logs in or out (including an expired token). */
export function onAuthChange(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function logout() {
  setToken(null);
}

// --- Requests ----------------------------------------------------------------

function errorMessage(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail.map((d) => (d as { msg?: string }).msg?.replace(/^Value error, /, "")).filter(Boolean);
    if (messages.length) return messages.join(" ");
  }
  if (status === 429) return "Too many requests. Try again shortly.";
  if (status >= 500) return "The server ran into a problem. Please try again.";
  return `Request failed (${status}).`;
}

async function send(path: string, init: RequestInit = {}, auth = true): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (auth && token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");

  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, { ...init, headers });
  } catch {
    throw new Error("Can't reach the LabLens server. Is the backend running?");
  }
  if (response.ok) return response;

  if (response.status === 401 && auth && token) setToken(null);
  const body = await response.json().catch(() => null);
  throw new Error(errorMessage(response.status, body));
}

async function json<T>(path: string, init?: RequestInit, auth = true): Promise<T> {
  const response = await send(path, init, auth);
  return (response.status === 204 ? undefined : await response.json()) as T;
}

const body = (data: unknown) => JSON.stringify(data);

export const api = {
  register: (name: string, email: string, password: string) =>
    json<User>("/auth/register", { method: "POST", body: body({ name, email, password }) }, false),

  async login(email: string, password: string) {
    const res = await json<{ access_token: string }>("/auth/login", { method: "POST", body: body({ email, password }) }, false);
    setToken(res.access_token);
  },

  me: () => json<User>("/auth/me"),

  listReports: () => json<ReportSummary[]>("/api/reports"),
  getReport: (id: string) => json<ReportDetail>(`/api/reports/${encodeURIComponent(id)}`),
  uploadReport(file: File) {
    const form = new FormData();
    form.append("file", file);
    return json<ReportDetail>("/api/reports", { method: "POST", body: form });
  },
  renameReport: (id: string, name: string) =>
    json<ReportDetail>(`/api/reports/${encodeURIComponent(id)}`, { method: "PATCH", body: body({ name }) }),
  deleteReport: (id: string) => json<void>(`/api/reports/${encodeURIComponent(id)}`, { method: "DELETE" }),
  deleteReportFile: (id: string) => json<ReportDetail>(`/api/reports/${encodeURIComponent(id)}/file`, { method: "DELETE" }),
  async getReportFile(id: string): Promise<Blob> {
    return (await send(`/api/reports/${encodeURIComponent(id)}/file`)).blob();
  },

  listChats: (reportId?: string) =>
    json<ChatSummary[]>(`/api/chats${reportId ? `?report_id=${encodeURIComponent(reportId)}` : ""}`),
  getChat: (id: string) => json<ChatDetail>(`/api/chats/${encodeURIComponent(id)}`),
  createChat: (reportId: string) => json<ChatDetail>("/api/chats", { method: "POST", body: body({ report_id: reportId }) }),
  renameChat: (id: string, title: string) =>
    json<ChatSummary>(`/api/chats/${encodeURIComponent(id)}`, { method: "PATCH", body: body({ title }) }),
  deleteChat: (id: string) => json<void>(`/api/chats/${encodeURIComponent(id)}`, { method: "DELETE" }),
  sendMessage: (chatId: string, question: string, testNames: string[]) =>
    json<SendMessageResponse>(`/api/chats/${encodeURIComponent(chatId)}/messages`, {
      method: "POST",
      body: body({ question, test_names: testNames }),
    }),

  getAudit: (taskId: string) => json<AuditEntry[]>(`/api/audit/${encodeURIComponent(taskId)}`),
};
