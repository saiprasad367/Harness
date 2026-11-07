/**
 * useBackendStatus.ts
 * Polls the FastAPI backend status every 3 seconds and exposes live metrics
 * via a WebSocket connection to /ws/metrics.
 */
import { useEffect, useRef, useState } from "react";
import {
  api,
  createMetricsSocket,
  isBackendReachable,
  type ApiStatus,
  type LiveMetrics,
} from "@/lib/api";

export interface BackendStatus {
  reachable: boolean;
  checking: boolean;
  status: ApiStatus | null;
  metrics: LiveMetrics | null;
  error: string | null;
}

export function useBackendStatus(): BackendStatus {
  const [reachable, setReachable] = useState(false);
  const [checking, setChecking] = useState(true);
  const [status, setStatus] = useState<ApiStatus | null>(null);
  const [metrics, setMetrics] = useState<LiveMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Poll status every 3s
  useEffect(() => {
    let alive = true;

    const check = async () => {
      setChecking(true);
      try {
        const ok = await isBackendReachable();
        if (!alive) return;
        setReachable(ok);
        if (ok) {
          const s = await api.getStatus();
          if (!alive) return;
          setStatus(s);
          setError(null);
        } else {
          setStatus(null);
          setError("Backend not running. Start api_server.py first.");
        }
      } catch (e) {
        if (!alive) return;
        setReachable(false);
        setError(String(e));
      } finally {
        if (alive) setChecking(false);
      }
    };

    check();
    const interval = setInterval(check, 3000);
    return () => {
      alive = false;
      clearInterval(interval);
    };
  }, []);

  // Metrics WebSocket
  useEffect(() => {
    if (!reachable) {
      wsRef.current?.close();
      wsRef.current = null;
      return;
    }
    if (wsRef.current) return; // already open

    const ws = createMetricsSocket(
      (m) => setMetrics(m),
      () => { wsRef.current = null; },
    );
    wsRef.current = ws;
    ws.onclose = () => { wsRef.current = null; };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [reachable]);

  return { reachable, checking, status, metrics, error };
}

