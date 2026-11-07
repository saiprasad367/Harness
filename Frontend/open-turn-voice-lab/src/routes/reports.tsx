import { createFileRoute } from "@tanstack/react-router";
import { ResearchPageContent } from "@/components/research/research-pages";
import { ResearchShell } from "@/components/research/research-shell";
import { pageHead } from "@/lib/research-data";

export const Route = createFileRoute("/reports")({
  head: () => pageHead("reports"),
  component: ReportsRoute,
});

function ReportsRoute() {
  return <ResearchShell page="reports"><ResearchPageContent page="reports" /></ResearchShell>;
}