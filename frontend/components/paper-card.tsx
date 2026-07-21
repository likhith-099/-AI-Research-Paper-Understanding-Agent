"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Copy, ChevronDown, ChevronUp } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

export function PaperCard({
  title,
  content,
  citations,
}: {
  title: string;
  content?: string;
  citations?: Array<{ section?: string; pages?: Array<string | number> }>;
}) {
  const [collapsed, setCollapsed] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content ?? "");
    toast.success(`${title} copied`);
  };

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={handleCopy}>
            <Copy className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setCollapsed((value) => !value)}>
            {collapsed ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
          </Button>
        </div>
      </CardHeader>
      <CardContent className={cn("space-y-3 text-sm leading-6 text-muted-foreground", collapsed && "hidden")}>
        <p>{content || "No content available."}</p>
        {citations?.length ? (
          <div className="flex flex-wrap gap-2 pt-2">
            {citations.map((citation, index) => (
              <span key={`${title}-citation-${index}`} className="rounded-full border border-border px-3 py-1 text-xs text-foreground/80">
                {citation.section ?? title} · p.{formatPages(citation.pages)}
              </span>
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function formatPages(pages?: Array<string | number>) {
  if (!pages || !pages.length) return "—";
  return pages.map(String).join(", ");
}
