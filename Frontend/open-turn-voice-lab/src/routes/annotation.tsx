import { createFileRoute } from "@tanstack/react-router";
import { ResearchPageContent } from "@/components/research/research-pages";
import { ResearchShell } from "@/components/research/research-shell";
import { pageHead } from "@/lib/research-data";

export const Route = createFileRoute("/annotation")({
  head: () => pageHead("annotation"),
  component: AnnotationRoute,
});

function AnnotationRoute() {
  return <ResearchShell page="annotation"><ResearchPageContent page="annotation" /></ResearchShell>;
}