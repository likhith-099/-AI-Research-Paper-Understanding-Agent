"use client";

import { PaperCard } from "@/components/paper-card";
import { RetrievalPanel } from "@/components/retrieval-panel";
import { ChatPanel } from "@/components/chat-panel";
import type { PaperAnalysis } from "@/types/paper";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const sections = [
  ["Summary", "summary"],
  ["Methodology", "methodology"],
  ["Datasets", "datasets"],
  ["Implementation", "implementation"],
  ["Contributions", "contributions"],
  ["Results", "results"],
  ["Limitations", "limitations"],
  ["Future Work", "future_work"],
  ["Equations", "equations"],
] as const;

export function Dashboard({ paper }: { paper: PaperAnalysis }) {
  const paperInfo = paper.paper_info ?? paper.debug_report?.paper_info ?? {};
  const sectionsFound = paper.debug_report?.detected_sections?.length ?? sections.length;

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Paper Information</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <Info label="Title" value={paperInfo.title ?? "Unavailable"} />
            <Info label="Authors" value={paperInfo.authors ?? "Unavailable"} />
            <Info label="Publication Year" value={paperInfo.year ?? "Unavailable"} />
            <Info label="Pages" value={paperInfo.pages ? String(paperInfo.pages) : "Unavailable"} />
            <Info label="Sections Found" value={String(sectionsFound)} />
          </CardContent>
        </Card>

        <div className="grid gap-4 md:grid-cols-2">
          {sections.map(([title, key]) => {
            const normalizedKey = key === "methodology" ? "method" : key === "datasets" ? "dataset" : key;
            return (
              <PaperCard
                key={title}
                title={title}
                content={(paper as Record<string, string | undefined>)[key] ?? paper.summary}
                citations={extractCitations(paper.debug_report?.section_traces?.[normalizedKey])}
              />
            );
          })}
        </div>

        <RetrievalPanel traces={paper.debug_report?.section_traces} />
      </div>

      <div className="space-y-6">
        <ChatPanel paper={paper} />
        <Card>
          <CardHeader>
            <CardTitle>Overview</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground">
            <div className="flex flex-wrap gap-2">
              <Badge>PDF Upload</Badge>
              <Badge>arXiv</Badge>
              <Badge>Hybrid Retrieval</Badge>
              <Badge>Groq</Badge>
            </div>
            <p>
              This dashboard surfaces the generated paper understanding output in a compact production-style layout.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Info({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border bg-accent p-4">
      <div className="text-xs uppercase tracking-[0.22em] text-muted-foreground">{label}</div>
      <div className="mt-2 text-sm font-medium">{value}</div>
    </div>
  );
}

function extractCitations(trace?: { retrieved_chunks?: Array<{ section?: string; pages?: Array<string | number> }> }) {
  const chunks = trace?.retrieved_chunks ?? [];
  return chunks.map((chunk) => ({
    section: chunk.section,
    pages: chunk.pages,
  }));
}
