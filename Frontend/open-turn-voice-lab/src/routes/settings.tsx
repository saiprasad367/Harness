import { createFileRoute } from "@tanstack/react-router";
import { ResearchPageContent } from "@/components/research/research-pages";
import { ResearchShell } from "@/components/research/research-shell";
import { pageHead } from "@/lib/research-data";

export const Route = createFileRoute("/settings")({
  head: () => pageHead("settings"),
  component: SettingsRoute,
});

function SettingsRoute() {
  return <ResearchShell page="settings"><ResearchPageContent page="settings" /></ResearchShell>;
}