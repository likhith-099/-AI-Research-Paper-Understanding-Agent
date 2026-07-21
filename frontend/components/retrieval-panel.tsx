"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { RetrievalTrace } from "@/types/paper";

export function RetrievalPanel({ traces }: { traces?: Record<string, RetrievalTrace> }) {
  const [open, setOpen] = useState(false);
  const firstTrace = traces ? Object.entries(traces)[0] : undefined;
  const trace = firstTrace?.[1];

  if (!trace) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Retrieval Details</CardTitle>
        <Button variant="ghost" size="sm" onClick={() => setOpen((value) => !value)}>
          {open ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </Button>
      </CardHeader>
      {open && (
        <CardContent className="grid gap-4 text-sm md:grid-cols-2">
          <Meta label="Retrieved Section" value={trace.retrieved_sections?.[0] ?? "—"} />
          <Meta
            label="Retrieved Parent ID"
            value={String(trace.retrieved_chunks?.[0]?.parent_id ?? trace.retrieved_ids?.[0] ?? "—")}
          />
          <Meta
            label="Retrieved Child ID"
            value={String(trace.retrieved_chunks?.[0]?.chunk_id ?? trace.retrieved_chunk_ids?.[0] ?? "—")}
          />
          <Meta
            label="Retrieved Pages"
            value={formatPages(trace.retrieved_chunks?.[0]?.pages ?? trace.chunks_sent_to_groq?.[0]?.pages)}
          />
          <Meta label="Cross Encoder Score" value={trace.rerank_scores?.[0]?.score?.toFixed(3) ?? "—"} />
          <Meta label="Fallback Mode" value={trace.fallback_mode ?? "—"} />
          <div className="md:col-span-2 flex flex-wrap gap-2">
            {trace.status && <Badge>{trace.status}</Badge>}
            {trace.context_length != null && <Badge>Context {trace.context_length}</Badge>}
            {trace.final_context_length != null && <Badge>Final {trace.final_context_length}</Badge>}
          </div>
        </CardContent>
      )}
    </Card>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border bg-accent p-4">
      <div className="text-xs uppercase tracking-[0.22em] text-muted-foreground">{label}</div>
      <div className="mt-2 font-medium">{value}</div>
    </div>
  );
}

function formatPages(value?: unknown) {
  if (!Array.isArray(value) || !value.length) return "—";
  return value.map(String).join(", ");
}
