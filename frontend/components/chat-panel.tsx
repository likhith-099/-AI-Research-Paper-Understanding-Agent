"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import type { ChatCitation, ChatMessage, PaperAnalysis } from "@/types/paper";

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const quickPrompts = [
  "What optimizer is used?",
  "What datasets are used?",
  "Explain Figure 3",
  "What are the limitations?",
];

export function ChatPanel({ paper }: { paper: PaperAnalysis | null }) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "assistant", content: "Ask me anything about the paper." },
  ]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);

  const send = async (question: string) => {
    const userMessage = question.trim();
    if (!userMessage) return;
    setIsSending(true);
    try {
      const response = await synthesizeAnswer(userMessage, paper);
      setMessages((current) => [
        ...current,
        { role: "user", content: userMessage },
        { role: "assistant", content: response.answer, citations: response.citations },
      ]);
      setInput("");
      toast.success("Answer generated");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Chat request failed";
      toast.error(message);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <Card className="sticky top-6">
      <CardHeader>
        <CardTitle>Chat With Paper</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="max-h-[420px] space-y-3 overflow-auto pr-1">
          {messages.map((message, index) => (
            <div
              key={`${message.role}-${index}`}
              className={`rounded-2xl px-4 py-3 text-sm leading-6 ${
                message.role === "user" ? "ml-8 bg-primary/15" : "mr-8 bg-accent"
              }`}
            >
              <div>{message.content}</div>
              {message.role === "assistant" && message.citations?.length ? (
                <div className="mt-3 flex flex-wrap gap-2">
                  {message.citations.map((citation, citationIndex) => (
                    <Badge key={`${citation.section}-${citation.page}-${citationIndex}`}>
                      {citation.section ?? "section"} · p.{citation.page ?? "?"}
                    </Badge>
                  ))}
                </div>
              ) : null}
            </div>
          ))}
        </div>
        <div className="flex flex-wrap gap-2">
          {quickPrompts.map((prompt) => (
            <Button key={prompt} variant="outline" size="sm" onClick={() => void send(prompt)} disabled={isSending}>
              {prompt}
            </Button>
          ))}
        </div>
        <div className="space-y-3">
          <Input
            placeholder="Ask a question about the paper..."
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && void send(input)}
          />
          <Button className="w-full" onClick={() => void send(input)} disabled={isSending}>
            {isSending ? "Thinking..." : "Send"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

async function synthesizeAnswer(question: string, paper: PaperAnalysis | null) {
  if (!paper) {
    return {
      answer: "Please run an analysis first so I can answer questions about the paper.",
      citations: [],
      routed_sections: [],
    };
  }

  const response = await fetch(`${apiBaseUrl}/chat-paper`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, paper }),
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? "Chat request failed");
  }

  return (await response.json()) as {
    answer: string;
    citations?: ChatCitation[];
    routed_sections?: string[];
  };
}
