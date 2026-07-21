"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

const steps = [
  "Document processing",
  "Extracting text",
  "Building retrieval pipeline",
  "Analyzing with AI",
];

export function LoadingState() {
  return (
    <Card className="overflow-hidden">
      <CardContent className="p-0">
        <div className="grid gap-0 md:grid-cols-[1.1fr_0.9fr]">
          <div className="border-b border-border p-8 md:border-b-0 md:border-r">
            <div className="mb-5 text-sm uppercase tracking-[0.3em] text-muted-foreground">Loading</div>
            <h2 className="text-2xl font-semibold">Preparing your paper</h2>
            <p className="mt-2 max-w-xl text-sm text-muted-foreground">
              We’re extracting the document structure, generating embeddings, and setting up the answer flow.
            </p>
            <div className="mt-8 space-y-4">
              {steps.map((step, index) => (
                <div key={step} className="flex items-center gap-4">
                  <div className="flex h-8 w-8 items-center justify-center rounded-full border border-border bg-accent text-sm">
                    {index + 1}
                  </div>
                  <div className="flex-1">
                    <div className="text-sm font-medium">{step}</div>
                    <Skeleton className="mt-2 h-2 w-full" />
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="flex items-center justify-center p-10">
            <div className="relative flex h-40 w-40 items-center justify-center">
              <div className="absolute inset-0 animate-pulse rounded-full bg-primary/15" />
              <div className="absolute inset-4 animate-ping rounded-full border border-primary/35" />
              <div className="relative flex h-24 w-24 items-center justify-center rounded-full border border-primary/30 bg-background shadow-soft">
                <div className="h-3 w-3 rounded-full bg-primary" />
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
