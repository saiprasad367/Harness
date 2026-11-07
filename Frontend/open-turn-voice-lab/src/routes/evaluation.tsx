import { createFileRoute } from "@tanstack/react-router";
import { ResearchPageContent } from "@/components/research/research-pages";
import { ResearchShell } from "@/components/research/research-shell";
import { pageHead } from "@/lib/research-data";

export const Route = createFileRoute("/evaluation")({
  head: () => pageHead("evaluation"),
  component: EvaluationRoute,
});

function EvaluationRoute() {
  return <ResearchShell page="evaluation"><ResearchPageContent page="evaluation" /></ResearchShell>;
}