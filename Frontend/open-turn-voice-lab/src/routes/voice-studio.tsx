import { createFileRoute } from "@tanstack/react-router";
import { ResearchPageContent } from "@/components/research/research-pages";
import { ResearchShell } from "@/components/research/research-shell";
import { pageHead } from "@/lib/research-data";

export const Route = createFileRoute("/voice-studio")({
  head: () => pageHead("voice"),
  component: VoiceStudioRoute,
});

function VoiceStudioRoute() {
  return <ResearchShell page="voice"><ResearchPageContent page="voice" /></ResearchShell>;
}