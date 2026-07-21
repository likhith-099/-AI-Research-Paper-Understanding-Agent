"use client";

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { LoadingState } from "@/components/loading-state";
import { Dashboard } from "@/components/dashboard";
import { toast } from "sonner";
import type { PaperAnalysis } from "@/types/paper";
import { Upload, Sparkles, MoonStar, SunMedium } from "lucide-react";
import { useTheme } from "@/components/theme-provider";

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export function Analyzer() {
  const { theme, toggleTheme } = useTheme();
  const [paperUrl, setPaperUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PaperAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = useMemo(() => Boolean(file || paperUrl.trim()), [file, paperUrl]);

  const submit = async () => {
    if (!canSubmit) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = file ? await analyzeUpload(file) : await analyzePaperUrl(paperUrl);
      setResult(response);
      toast.success("Analysis complete");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Analysis failed";
      setError(message);
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen px-4 py-6 md:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <header className="glass flex items-center justify-between rounded-3xl border border-border px-5 py-4 shadow-soft">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-primary text-primary-foreground">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <div className="text-lg font-semibold">PaperLens AI</div>
              <div className="text-sm text-muted-foreground">Research paper understanding agent</div>
            </div>
          </div>
          <Button variant="outline" onClick={toggleTheme}>
            {theme === "dark" ? <SunMedium className="h-4 w-4" /> : <MoonStar className="h-4 w-4" />}
            {theme === "dark" ? "Light mode" : "Dark mode"}
          </Button>
        </header>

        {!result && !isLoading && (
          <div className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
            <Card>
              <CardHeader>
                <CardTitle>Analyze a research paper</CardTitle>
              </CardHeader>
              <CardContent className="space-y-5">
                <div className="space-y-2">
                  <label className="text-sm font-medium">ArXiv URL</label>
                  <Input placeholder="https://arxiv.org/abs/..." value={paperUrl} onChange={(event) => setPaperUrl(event.target.value)} />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Paste arXiv URL or upload PDF</label>
                  <div
                    className="rounded-2xl border border-dashed border-border bg-accent/30 p-6 text-center"
                    onDragOver={(event) => event.preventDefault()}
                    onDrop={(event) => {
                      event.preventDefault();
                      const dropped = event.dataTransfer.files?.[0];
                      if (dropped) setFile(dropped);
                    }}
                  >
                    <Upload className="mx-auto h-6 w-6 text-muted-foreground" />
                    <p className="mt-3 text-sm text-muted-foreground">Drag and drop a PDF here, or choose one below.</p>
                    <Input
                      type="file"
                      accept="application/pdf"
                      className="mt-4"
                      onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                    />
                    {file && <Badge className="mt-4">{file.name}</Badge>}
                  </div>
                </div>
                <div className="flex flex-wrap gap-3">
                  <Button size="lg" onClick={submit} disabled={!canSubmit}>
                    Analyze
                  </Button>
                  <Button variant="outline" size="lg" onClick={() => { setFile(null); setPaperUrl(""); }}>
                    Reset
                  </Button>
                </div>
                {error && <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-200">{error}</div>}
              </CardContent>
            </Card>

            <Card className="overflow-hidden">
              <CardHeader>
                <CardTitle>What you’ll get</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-3 sm:grid-cols-2">
                {["Summary", "Methodology", "Datasets", "Implementation", "Contributions", "Results", "Limitations", "Future Work", "Equations"].map((item) => (
                  <div key={item} className="rounded-2xl border border-border bg-accent p-4 text-sm font-medium">
                    {item}
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        )}

        {isLoading && <LoadingState />}
        {result && <Dashboard paper={normalizePaper(result)} />}
      </div>
    </div>
  );
}

function normalizePaper(paper: PaperAnalysis): PaperAnalysis {
  return {
    ...paper,
    methodology: paper.methodology ?? paper.method,
    datasets: paper.datasets ?? paper.dataset,
  };
}

async function analyzePaperUrl(paperUrl: string): Promise<PaperAnalysis> {
  const response = await fetch(`${apiBaseUrl}/analyze-paper`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paper_url: paperUrl }),
  });
  if (!response.ok) throw new Error(await extractError(response));
  return response.json();
}

async function analyzeUpload(file: File): Promise<PaperAnalysis> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${apiBaseUrl}/analyze-upload`, { method: "POST", body: formData });
  if (!response.ok) throw new Error(await extractError(response));
  return response.json();
}

async function extractError(response: Response) {
  try {
    const payload = await response.json();
    return payload.detail ?? response.statusText;
  } catch {
    return response.statusText;
  }
}
