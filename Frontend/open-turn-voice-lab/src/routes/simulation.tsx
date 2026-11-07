import { createFileRoute } from "@tanstack/react-router";
import { ResearchPageContent } from "@/components/research/research-pages";
import { ResearchShell } from "@/components/research/research-shell";
import { pageHead } from "@/lib/research-data";

export const Route = createFileRoute("/simulation")({
  head: () => pageHead("simulation"),
  component: SimulationRoute,
});

function SimulationRoute() {
  return <ResearchShell page="simulation"><ResearchPageContent page="simulation" /></ResearchShell>;
}