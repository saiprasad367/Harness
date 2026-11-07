/**
 * useApiData.ts — React Query hooks for all REST endpoints
 * Drop-in replacements for every static import from research-data.ts
 */
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

// ── Status ────────────────────────────────────────────────────────────────────
export function useStatus() {
  return useQuery({
    queryKey: ["status"],
    queryFn: api.getStatus,
    refetchInterval: 3000,
    retry: false,
  });
}

// ── Dataset ───────────────────────────────────────────────────────────────────
export function useDatasetStats() {
  return useQuery({
    queryKey: ["dataset", "stats"],
    queryFn: api.getDatasetStats,
    refetchInterval: 60_000,
    staleTime: 30_000,
  });
}

export function useSpeakers() {
  return useQuery({
    queryKey: ["dataset", "speakers"],
    queryFn: api.getSpeakers,
    staleTime: 60_000,
  });
}

export function useRecordings(speakerId: string | null) {
  return useQuery({
    queryKey: ["dataset", "recordings", speakerId],
    queryFn: () => api.getRecordings(speakerId!),
    enabled: !!speakerId,
    staleTime: 30_000,
  });
}

// ── Evaluation ────────────────────────────────────────────────────────────────
export function useBenchmarks() {
  return useQuery({
    queryKey: ["evaluation", "benchmarks"],
    queryFn: api.getBenchmarks,
    staleTime: 60_000,
  });
}

export function useEvaluationResults() {
  return useQuery({
    queryKey: ["evaluation", "results"],
    queryFn: api.getEvaluationResults,
    staleTime: 60_000,
  });
}

// ── Simulation ────────────────────────────────────────────────────────────────
export function useSimulation(mode: string, speaker?: string, enabled = false) {
  return useQuery({
    queryKey: ["simulation", mode, speaker],
    queryFn: () => api.runSimulation(mode, speaker),
    enabled,
    staleTime: 10_000,
  });
}

// ── Annotation ────────────────────────────────────────────────────────────────
export function useAnnotationQueue() {
  return useQuery({
    queryKey: ["annotation", "queue"],
    queryFn: api.getAnnotationQueue,
    refetchInterval: 5000,
  });
}

export function useAnnotationStats() {
  return useQuery({
    queryKey: ["annotation", "stats"],
    queryFn: api.getAnnotationStats,
    refetchInterval: 5000,
  });
}

export function useSubmitAnnotationLabel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.submitAnnotationLabel,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["annotation"] });
    },
  });
}

// ── Reports ───────────────────────────────────────────────────────────────────
export function useReportsSummary() {
  return useQuery({
    queryKey: ["reports", "summary"],
    queryFn: api.getReportsSummary,
    staleTime: 30_000,
  });
}

// ── Settings / Devices ────────────────────────────────────────────────────────
export function useConfig() {
  return useQuery({
    queryKey: ["settings", "config"],
    queryFn: api.getConfig,
    staleTime: 10_000,
  });
}

export function useMicrophones() {
  return useQuery({
    queryKey: ["devices", "microphones"],
    queryFn: api.getMicrophones,
    staleTime: 60_000,
  });
}

export function useSpeakerDevices() {
  return useQuery({
    queryKey: ["devices", "speakers"],
    queryFn: api.getSpeakerDevices,
    staleTime: 60_000,
  });
}

export function useUpdateMoshiUrl() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (url: string) => api.updateMoshiUrl(url),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["settings"] });
      void qc.invalidateQueries({ queryKey: ["status"] });
    },
  });
}

