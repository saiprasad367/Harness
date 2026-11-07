import { createFileRoute } from "@tanstack/react-router";
import { ResearchShell } from "@/components/research/research-shell";
import { ResearchPageContent } from "@/components/research/research-pages";
import { pageHead } from "@/lib/research-data";

export const Route = createFileRoute("/")({
  head: () => pageHead("overview"),
  component: Index,
});

function Index() {
  return <ResearchShell page="overview"><ResearchPageContent page="overview" /></ResearchShell>;
}
