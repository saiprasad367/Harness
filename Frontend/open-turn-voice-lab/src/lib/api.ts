/**
 * api.ts — All API calls to the FastAPI backend at localhost:8000
 * Zero mock data. Real-time REST + WebSocket.
 */

export const API_BASE =
  (import.meta as any).env?.VITE_API_BASE_URL ||
  (typeof window !== "undefined" && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1"
    ? `${window.location.protocol}//${window.location.host}`
    : "http://localhost:8000");

export const WS_BASE =
  (import.meta as any).env?.VITE_WS_BASE_URL ||
  API_BASE.replace(/^http:\/\//i, "ws://").replace(/^https:\/\//i, "wss://");

// ── Types ────────────────────────────────────────────────────────────────────

export interface ApiStatus {
  moshi_url: string;
  moshi_connected: boolean;
  moshi_connecting: boolean;
  session_active: boolean;
  session_elapsed: number;
  vad_available: boolean;
  classifier_available: boolean;
  controller_available: boolean;
  dataset_available: boolean;
  eval_available: boolean;
  pipeline: PipelineNode[];
}

export interface PipelineNode {
  name: string;
  stage: string;
  status: string;
  latency: string;
}

export interface Speaker {
  id: string;
  split: string;
  cohort: string;
  gender: string;
  recordings: number;
}

export interface DatasetStats {
  total_recordings: number;
  total_speakers: number;
  splits: { train: number; val: number; test: number };
  cohorts: { control: number; dysarthric: number };
  sample_rate_hz: number;
  mic_type: string;
  speakers: Speaker[];
  split_percentages: { train: number; val: number; test: number };
}

export interface Recording {
  id: string;
  file: string;
  duration: string;
  seconds: number;
  format: string;
  sample_rate: string;
  status: string;
  pauses: number;
}

export interface BenchmarkMetric {
  name: string;
  value: string;
  compare: string;
  definition: string;
}

export interface BenchmarkResult {
  run_id: string;
  model: string;
  dataset: string;
  date: string;
  metrics: BenchmarkMetric[];
  confusion_matrix: number[][];
  performance_trend: { run: string; adaptive: number; baseline: number }[];
  decision_distribution: { name: string; value: number }[];
}

export interface CandidatePause {
  id: string;
  recording: string;
  speaker: string;
  cohort: string;
  duration: string;
  confidence: number;
  priority: string;
  labeled?: boolean;
  label?: string;
}

export interface AnnotationQueue {
  candidates: CandidatePause[];
  total: number;
  labeled: number;
  remaining: number;
}

export interface AnnotationStats {
  total_candidates: number;
  labeled: number;
  remaining: number;
  progress_pct: number;
  distribution: { END: number; CONTINUE: number; UNCERTAIN: number };
}

export interface SimulationResult {
  mode: string;
  speaker: string;
  events: { time: string; event: string; detail: string }[];
  comparison: {
    adaptive: { END: number; CONTINUE: number; UNCERTAIN: number; false_interruptions: number; delayed_responses: number; avg_response_delay_s: number; threshold: number };
    fixed: { END: number; CONTINUE: number; UNCERTAIN: number; false_interruptions: number; delayed_responses: number; avg_response_delay_s: number; threshold: number };
  };
  threshold_trace: { t: number; adaptive: number; fixed: number }[];
  recording_id: string;
  duration: string;
}

export interface ReportSummary {
  reports: {
    id: string;
    title: string;
    type: string;
    date: string;
    status: string;
    summary: string;
  }[];
  overview: {
    total_recordings: number;
    total_speakers: number;
    annotation_progress_pct: number;
    latest_eval_accuracy: string;
  };
}

export interface AudioDevice {
  id: string;
  name: string;
  is_default: boolean;
  channels?: number;
  sample_rate?: number;
}

export interface LiveMetrics {
  type: string;
  ts: number;
  moshi_connected: boolean;
  session_active: boolean;
  vad_state: string;
  speech_prob: number;
  pause_duration: number;
  adaptive_intention: string;
  p_end: number;
  controller_state: string;
  controller_action: string;
  mic_level_db: number;
  vad_latency_ms: number;
  decision_latency_ms: number;
  audio_buffer_chunks: number;
  dropped_frames: number;
  captured_chunks: number;
  session_elapsed: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} → ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ── REST API ─────────────────────────────────────────────────────────────────

export const api = {
  getStatus: () => apiFetch<ApiStatus>("/api/status"),

  getDatasetStats: () => apiFetch<DatasetStats>("/api/dataset/stats"),
  getSpeakers: () => apiFetch<{ speakers: Speaker[] }>("/api/dataset/speakers"),
  getRecordings: (speakerId: string) =>
    apiFetch<{ speaker_id: string; recordings: Recording[] }>(`/api/dataset/recordings/${speakerId}`),

  getBenchmarks: () => apiFetch<BenchmarkResult>("/api/evaluation/benchmarks"),
  getEvaluationResults: () => apiFetch<{ runs: unknown[] }>("/api/evaluation/results"),

  runSimulation: (mode: string, speaker?: string) =>
    apiFetch<SimulationResult>(`/api/simulation/run?mode=${encodeURIComponent(mode)}${speaker ? `&speaker=${encodeURIComponent(speaker)}` : ""}`),

  getAnnotationQueue: () => apiFetch<AnnotationQueue>("/api/annotation/queue"),
  submitAnnotationLabel: (payload: {
    pause_id: string;
    label: string;
    confidence: string;
    notes: string;
  }) =>
    apiFetch<{ success: boolean; total_labeled: number }>("/api/annotation/label", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getAnnotationStats: () => apiFetch<AnnotationStats>("/api/annotation/stats"),

  getReportsSummary: () => apiFetch<ReportSummary>("/api/reports/summary"),

  getConfig: () => apiFetch<{ moshi_url: string; moshi_connected: boolean; version: string }>("/api/settings/config"),
  updateMoshiUrl: (url: string) =>
    apiFetch<{ success: boolean; url: string }>("/api/settings/moshi-url", {
      method: "POST",
      body: JSON.stringify({ url }),
    }),

  getMicrophones: () => apiFetch<{ devices: AudioDevice[] }>("/api/devices/microphones"),
  getSpeakerDevices: () => apiFetch<{ devices: AudioDevice[] }>("/api/devices/speakers"),

  startSession: (mode: string) =>
    apiFetch<{ success: boolean; session_start: number }>("/api/session/start", {
      method: "POST",
      body: JSON.stringify({ mode }),
    }),
  stopSession: () =>
    apiFetch<{ success: boolean }>("/api/session/stop", { method: "POST", body: "{}" }),

  connectMoshi: () =>
    apiFetch<{ success: boolean; url: string }>("/api/moshi/connect", { method: "POST", body: "{}" }),
  disconnectMoshi: () =>
    apiFetch<{ success: boolean }>("/api/moshi/disconnect", { method: "POST", body: "{}" }),
};

// ── WebSocket helpers ─────────────────────────────────────────────────────────

export function createMetricsSocket(
  onMetrics: (m: LiveMetrics) => void,
  onError?: (e: Event) => void,
): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/ws/metrics`);
  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data as string) as LiveMetrics;
      if (data.type === "metrics") onMetrics(data);
    } catch {
      // ignore
    }
  };
  if (onError) ws.onerror = onError;
  // Keepalive ping every 20s
  const ping = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send("ping");
  }, 20_000);
  ws.onclose = () => clearInterval(ping);
  return ws;
}

export function createVoiceSocket(
  onMessage: (msg: Record<string, unknown>) => void,
  onError?: (e: Event) => void,
): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/ws/voice`);
  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data as string) as Record<string, unknown>;
      onMessage(data);
    } catch {
      // ignore
    }
  };
  if (onError) ws.onerror = onError;
  return ws;
}

export function createMoshiBridgeSocket(
  onMessage: (msg: Record<string, unknown>) => void,
  onError?: (e: Event) => void,
): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/ws/moshi`);
  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data as string) as Record<string, unknown>;
      onMessage(data);
    } catch {
      // ignore
    }
  };
  if (onError) ws.onerror = onError;
  return ws;
}

// ── Backend reachability check ────────────────────────────────────────────────
export async function isBackendReachable(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/status`, { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch {
    return false;
  }
}

