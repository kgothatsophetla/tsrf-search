"use client";

import { useEffect, useRef, useState } from "react";
import ChatInput from "@/components/ChatInput";
import ChatMessage from "@/components/ChatMessage";
import type { AssistantMessage, Message, SearchResult } from "@/lib/types";

const SUGGESTED = [
  "What is TSRF?",
  "How do I submit a claim?",
  "What documents do I need for a death claim?",
  "How do I check my fund balance?",
];

function parseSSEBuffer(buffer: string): { events: unknown[]; remainder: string } {
  const lines = buffer.split("\n");
  const remainder = lines.pop() ?? "";
  const events: unknown[] = [];
  for (const line of lines) {
    if (!line.startsWith("data: ")) continue;
    try {
      events.push(JSON.parse(line.slice(6)));
    } catch {
      // skip malformed lines
    }
  }
  return { events, remainder };
}

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function updateAssistant(id: string, updater: (m: AssistantMessage) => AssistantMessage) {
    setMessages((prev) =>
      prev.map((m) => (m.id === id && m.role === "assistant" ? updater(m) : m))
    );
  }

  async function handleSend(text: string) {
    if (isStreaming) return;

    const userId = crypto.randomUUID();
    const assistantId = crypto.randomUUID();

    setMessages((prev) => [
      ...prev,
      { id: userId, role: "user", text },
      { id: assistantId, role: "assistant", results: [], status: "loading" },
    ]);

    setIsStreaming(true);

    try {
      const response = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, top_k: 5 }),
      });

      if (!response.ok || !response.body) {
        updateAssistant(assistantId, (m) => ({
          ...m,
          status: "error",
          error: "Search request failed. Is the API server running?",
        }));
        return;
      }

      updateAssistant(assistantId, (m) => ({ ...m, status: "streaming" }));

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const { events, remainder } = parseSSEBuffer(buffer);
        buffer = remainder;

        for (const event of events) {
          const e = event as Record<string, unknown>;
          if (e.type === "result") {
            updateAssistant(assistantId, (m) => ({
              ...m,
              results: [...m.results, e as unknown as SearchResult],
            }));
          } else if (e.type === "done") {
            updateAssistant(assistantId, (m) => ({ ...m, status: "done" }));
          } else if (e.type === "no_knowledge") {
            updateAssistant(assistantId, (m) => ({ ...m, status: "no_knowledge" }));
          } else if (e.type === "error") {
            updateAssistant(assistantId, (m) => ({
              ...m,
              status: "error",
              error: String(e.message ?? "Search failed"),
            }));
          }
        }
      }

      updateAssistant(assistantId, (m) =>
        m.status === "streaming" ? { ...m, status: "done" } : m
      );
    } catch (e) {
      updateAssistant(assistantId, (m) => ({
        ...m,
        status: "error",
        error: e instanceof Error ? e.message : "An unexpected error occurred",
      }));
    } finally {
      setIsStreaming(false);
    }
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <header className="flex-shrink-0 bg-white border-b border-gray-200 px-4 py-4">
        <div className="max-w-3xl mx-auto flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="white"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.35-4.35" />
            </svg>
          </div>
          <div>
            <h1 className="text-sm font-semibold text-gray-900">TSRF Knowledge Base</h1>
            <p className="text-xs text-gray-500">Semantic search over the TSRF FAQ</p>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center min-h-[60vh] text-center gap-6">
              <div>
                <div className="w-14 h-14 rounded-full bg-blue-50 flex items-center justify-center mx-auto mb-4">
                  <svg
                    width="22"
                    height="22"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="#2563eb"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <circle cx="11" cy="11" r="8" />
                    <path d="m21 21-4.35-4.35" />
                  </svg>
                </div>
                <h2 className="text-lg font-semibold text-gray-800">
                  Ask anything about TSRF
                </h2>
                <p className="text-sm text-gray-400 mt-1 max-w-sm">
                  Questions are matched semantically against the TSRF FAQ document.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
                {SUGGESTED.map((q) => (
                  <button
                    key={q}
                    onClick={() => handleSend(q)}
                    disabled={isStreaming}
                    className="text-left text-sm text-gray-600 bg-white border border-gray-200 rounded-xl px-4 py-3 hover:border-blue-400 hover:text-blue-600 transition-colors disabled:opacity-50"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </main>

      <ChatInput onSend={handleSend} disabled={isStreaming} />
    </div>
  );
}
