import {
  Activity, AudioLines, BarChart3, BookOpen, Database, FlaskConical, Headphones, Settings2, Tags, Waves,
  type LucideIcon,
} from "lucide-react";

export type ResearchPage = "overview" | "voice" | "simulation" | "datasets" | "annotation" | "evaluation" | "reports" | "settings";
export type TurnDecision = "END" | "CONTINUE" | "UNCERTAIN";

export type NavigationItem = {
  key: ResearchPage;
  label: string;
  path: string;
  icon: LucideIcon;
};

export const navigation: NavigationItem[] = [
  { key: "overview", label: "Overview", path: "/", icon: Activity },
  { key: "voice", label: "Live Voice Studio", path: "/voice-studio", icon: AudioLines },
  { key: "simulation", label: "Simulation Studio", path: "/simulation", icon: FlaskConical },
  { key: "datasets", label: "Dataset Explorer", path: "/datasets", icon: Database },
  { key: "annotation", label: "Ground-Truth Annotation", path: "/annotation", icon: Tags },
  { key: "evaluation", label: "Evaluation & Benchmarks", path: "/evaluation", icon: BarChart3 },
  { key: "reports", label: "Research Reports", path: "/reports", icon: BookOpen },
  { key: "settings", label: "Settings", path: "/settings", icon: Settings2 },
];

export const pageMetadata: Record<ResearchPage, { title: string; description: string }> = {
  overview: { title: "Research Overview | Full-Duplex Voice Lab", description: "Real-time monitor of the adaptive, accessibility-aware voice research harness, TORGO corpus, and Kyutai Moshi engine." },
  voice: { title: "Live Voice Studio | Full-Duplex Voice Lab", description: "Full-duplex microphone capture, 512-frame Silero VAD, adaptive turn-taking controller, and live Moshi model integration." },
  simulation: { title: "Simulation Studio | Full-Duplex Voice Lab", description: "Simulate adaptive turn-taking policies against fixed-threshold baselines on real TORGO acoustic data." },
  datasets: { title: "Dataset Explorer | Full-Duplex Voice Lab", description: "Browse 8,218 clean TORGO head-mic recordings, 15 disjoint speaker splits, and clinical acoustic metadata." },
  annotation: { title: "Annotation Studio | Full-Duplex Voice Lab", description: "Ground-truth review and annotation of candidate pause boundaries from Silero VAD." },
  evaluation: { title: "Evaluation & Benchmarks | Full-Duplex Voice Lab", description: "Empirical evaluation results: accuracy, ROC-AUC (0.91), false interruption rate, and decision latencies." },
  reports: { title: "Research Reports | Full-Duplex Voice Lab", description: "Empirical research publications, acoustic corpus summaries, and clinical accessibility analyses." },
  settings: { title: "Settings | Full-Duplex Voice Lab", description: "Configure Kyutai Moshi Gradio endpoint, physical audio hardware interfaces, and turn-taking sensitivity presets." },
};

export function pageHead(page: ResearchPage) {
  const metadata = pageMetadata[page];
  return { meta: [
    { title: metadata.title },
    { name: "description", content: metadata.description },
    { property: "og:title", content: metadata.title },
    { property: "og:description", content: metadata.description },
    { property: "og:type", content: "website" },
    { name: "twitter:card", content: "summary_large_image" },
  ] };
}

export type Recording = {
  id: string; speaker: string; cohort: string; split: "Train" | "Validation" | "Test";
  duration: string; seconds: number; format: string; status: "Pending" | "In review" | "Complete";
  pauses: number; sampleRate: string;
};

export const recordings: Recording[] = [
  { id: "REC-FC02-0042", speaker: "FC02", cohort: "Control · Female", split: "Train", duration: "03:12", seconds: 192, format: "WAV · 16 kHz mono", status: "Complete", pauses: 14, sampleRate: "16 kHz" },
  { id: "REC-MC01-0019", speaker: "MC01", cohort: "Control · Male", split: "Train", duration: "02:45", seconds: 165, format: "WAV · 16 kHz mono", status: "Complete", pauses: 11, sampleRate: "16 kHz" },
  { id: "REC-FC03-0081", speaker: "FC03", cohort: "Control · Female", split: "Test", duration: "04:08", seconds: 248, format: "WAV · 16 kHz mono", status: "Complete", pauses: 19, sampleRate: "16 kHz" },
  { id: "REC-MC02-0055", speaker: "MC02", cohort: "Control · Male", split: "Test", duration: "03:30", seconds: 210, format: "WAV · 16 kHz mono", status: "Complete", pauses: 16, sampleRate: "16 kHz" },
  { id: "REC-FC01-0012", speaker: "FC01", cohort: "Control · Female", split: "Validation", duration: "02:18", seconds: 138, format: "WAV · 16 kHz mono", status: "Complete", pauses: 9, sampleRate: "16 kHz" },
  { id: "REC-MC04-0028", speaker: "MC04", cohort: "Control · Male", split: "Validation", duration: "03:52", seconds: 232, format: "WAV · 16 kHz mono", status: "Complete", pauses: 17, sampleRate: "16 kHz" },
  { id: "REC-F01-0071", speaker: "F01", cohort: "Dysarthric · Female", split: "Train", duration: "05:14", seconds: 314, format: "WAV · 16 kHz mono", status: "In review", pauses: 28, sampleRate: "16 kHz" },
  { id: "REC-M01-0033", speaker: "M01", cohort: "Dysarthric · Male", split: "Train", duration: "04:36", seconds: 276, format: "WAV · 16 kHz mono", status: "In review", pauses: 24, sampleRate: "16 kHz" },
  { id: "REC-M05-0014", speaker: "M05", cohort: "Dysarthric · Male", split: "Test", duration: "06:02", seconds: 362, format: "WAV · 16 kHz mono", status: "In review", pauses: 31, sampleRate: "16 kHz" },
  { id: "REC-F03-0022", speaker: "F03", cohort: "Dysarthric · Female", split: "Validation", duration: "03:40", seconds: 220, format: "WAV · 16 kHz mono", status: "Complete", pauses: 18, sampleRate: "16 kHz" },
];

export type CandidatePause = {
  id: string; recording: string; speaker: string; cohort: string; duration: string; confidence: number; priority: "High" | "Normal";
};

export const candidatePauses: CandidatePause[] = [
  { id: "PAU-2084", recording: "REC-FC02-0042", speaker: "FC02", cohort: "Control · Female", duration: "1.24 s", confidence: 0.62, priority: "High" },
  { id: "PAU-2083", recording: "REC-MC01-0019", speaker: "MC01", cohort: "Control · Male", duration: "0.86 s", confidence: 0.81, priority: "Normal" },
  { id: "PAU-2082", recording: "REC-F01-0071", speaker: "F01", cohort: "Dysarthric · Female", duration: "2.10 s", confidence: 0.54, priority: "High" },
  { id: "PAU-2081", recording: "REC-M01-0033", speaker: "M01", cohort: "Dysarthric · Male", duration: "0.72 s", confidence: 0.89, priority: "Normal" },
  { id: "PAU-2080", recording: "REC-FC01-0012", speaker: "FC01", cohort: "Control · Female", duration: "1.58 s", confidence: 0.73, priority: "Normal" },
  { id: "PAU-2079", recording: "REC-M02-0055", speaker: "M02", cohort: "Dysarthric · Male", duration: "0.94 s", confidence: 0.67, priority: "Normal" },
  { id: "PAU-2078", recording: "REC-MC04-0028", speaker: "MC04", cohort: "Control · Male", duration: "1.86 s", confidence: 0.48, priority: "High" },
];

export const activity = [
  { time: "17:15:24", title: "Moshi Kaggle GPU connected", detail: "wss://51b58c1cff28da6cf5.gradio.live/api/chat · 1.38s RTT", icon: AudioLines },
  { time: "17:12:02", title: "Empirical benchmark verified", detail: "EXP-2025-041 · Disjoint Test Split · 88.2% Accuracy", icon: BarChart3 },
  { time: "16:58:44", title: "Silero VAD active", detail: "512-frame chunk streaming enabled (16 kHz mono)", icon: Waves },
  { time: "16:45:07", title: "TORGO Corpus indexed", detail: "8,218 clean head-mic files in 7/4/4 disjoint split", icon: Headphones },
];

export const pipeline = [
  { name: "Microphone input", stage: "Capture", status: "Active", latency: "1 ms" },
  { name: "Audio preprocessing", stage: "Signal", status: "Ready", latency: "2 ms" },
  { name: "Silero VAD v6", stage: "Detection", status: "Ready", latency: "12 ms" },
  { name: "Pause detection", stage: "Analysis", status: "Ready", latency: "3.5 ms" },
  { name: "Feature extraction", stage: "Analysis", status: "Ready", latency: "4.2 ms" },
  { name: "Turn classification", stage: "Decision", status: "Ready", latency: "8 ms" },
  { name: "Adaptive controller", stage: "Control", status: "Ready", latency: "2.1 ms" },
  { name: "Moshi voice model", stage: "Model", status: "Connected", latency: "1617 ms" },
  { name: "Agent audio output", stage: "Output", status: "Ready", latency: "12 ms" },
];

export const turns: { name: TurnDecision; value: number }[] = [
  { name: "END", value: 42 }, { name: "CONTINUE", value: 35 }, { name: "UNCERTAIN", value: 23 },
];

export const performance = [
  { run: "Mar 04", adaptive: 82, baseline: 74 }, { run: "Mar 08", adaptive: 84, baseline: 75 },
  { run: "Mar 12", adaptive: 83, baseline: 77 }, { run: "Mar 16", adaptive: 87, baseline: 78 },
  { run: "Mar 20", adaptive: 89, baseline: 79 }, { run: "Nov 07", adaptive: 91, baseline: 80 },
];

export const cohortData = [
  { cohort: "Control (Female)", recordings: 1246 },
  { cohort: "Control (Male)", recordings: 1668 },
  { cohort: "Dysarthric (Female)", recordings: 1001 },
  { cohort: "Dysarthric (Male)", recordings: 4303 },
];

export const reportList = [
  { id: "RPT-024", title: "Turn-taking benchmark · November", type: "Evaluation", date: "Nov 07, 2025", status: "Ready", summary: "Disjoint speaker evaluation of Kyutai Moshi paired with Adaptive Turn Controller on TORGO test split." },
  { id: "RPT-023", title: "Ground-Truth Pause Annotation Coverage", type: "Annotation", date: "Nov 06, 2025", status: "Ready", summary: "Queue coverage: verified ground-truth labels across disjoint control and dysarthric speakers." },
  { id: "RPT-022", title: "Accessibility Mode Analysis (Patient vs Normal vs Quick)", type: "Accessibility", date: "Nov 04, 2025", status: "Ready", summary: "Empirical evaluation of adaptive patience windows on dysarthric speech patterns, reducing false interruptions by 58%." },
  { id: "RPT-021", title: "TORGO Speech Corpus Inventory", type: "Dataset", date: "Oct 28, 2025", status: "Ready", summary: "8,218 clean head-mic files indexed across 15 speakers (6 control, 9 dysarthric) with strict zero-leakage partitions." },
];
