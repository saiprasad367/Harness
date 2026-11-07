import { createFileRoute } from "@tanstack/react-router";
import { ResearchPageContent } from "@/components/research/research-pages";
import { ResearchShell } from "@/components/research/research-shell";
import { pageHead } from "@/lib/research-data";

export const Route = createFileRoute("/datasets")({
  head: () => pageHead("datasets"),
  component: DatasetsRoute,
});

function DatasetsRoute() {
  return <ResearchShell page="datasets"><ResearchPageContent page="datasets" /></ResearchShell>;
}