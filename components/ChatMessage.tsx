"use client";

import { useState } from "react";
import type { AssistantMessage, Message, SearchResult, UserMessage } from "@/lib/types";

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 px-1 h-8">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
          style={{ animationDelay: `${i * 150}ms` }}
        />
      ))}
    </div>
  );
}

function SourceCard({ result, position }: { result: SearchResult; position: number }) {
  const pct = Math.round(result.score * 100);
  const badge =
    result.score >= 0.75
      ? "bg-green-50 text-green-700"
      : result.score >= 0.55
      ? "bg-yellow-50 text-yellow-700"
      : "bg-gray-100 text-gray-600";

  return (
    <div key={position} className="border border-gray-200 rounded-xl p-4 bg-white animate-slide-in">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-widest">
          Source {result.index + 1}
        </span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">Page {result.page}</span>
          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${badge}`}>
            {pct}% match
          </span>
        </div>
      </div>
      <p className="text-sm text-gray-700 leading-relaxed">{result.text}</p>
      <p className="mt-3 text-xs text-gray-400 truncate">{result.source}</p>
    </div>
  );
}

function SourceCarousel({
  results,
  isStreaming,
}: {
  results: SearchResult[];
  isStreaming: boolean;
}) {
  const [index, setIndex] = useState(0);

  if (results.length === 0) return null;

  const safeIndex = Math.min(index, results.length - 1);
  const canPrev = safeIndex > 0;
  const canNext = safeIndex < results.length - 1;

  return (
    <div className="space-y-3">
      <SourceCard key={safeIndex} result={results[safeIndex]} position={safeIndex} />

      <div className="flex items-center justify-between px-1">
        <button
          onClick={() => setIndex((i) => Math.max(0, i - 1))}
          disabled={!canPrev}
          className="w-8 h-8 rounded-full border border-gray-200 bg-white flex items-center justify-center hover:border-blue-400 hover:text-blue-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M15 18l-6-6 6-6" />
          </svg>
        </button>

        <div className="flex items-center gap-1.5">
          {results.map((_, i) => (
            <button
              key={i}
              onClick={() => setIndex(i)}
              className={`rounded-full transition-all ${
                i === safeIndex
                  ? "w-4 h-2 bg-blue-600"
                  : "w-2 h-2 bg-gray-300 hover:bg-gray-400"
              }`}
            />
          ))}
          {isStreaming && (
            <span className="w-2 h-2 rounded-full bg-gray-200 animate-pulse" />
          )}
        </div>

        <button
          onClick={() => setIndex((i) => Math.min(results.length - 1, i + 1))}
          disabled={!canNext}
          className="w-8 h-8 rounded-full border border-gray-200 bg-white flex items-center justify-center hover:border-blue-400 hover:text-blue-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9 18l6-6-6-6" />
          </svg>
        </button>
      </div>

      <p className="text-center text-xs text-gray-400">
        {safeIndex + 1} of {results.length}{isStreaming ? "+" : ""}
      </p>
    </div>
  );
}

function AssistantAvatar() {
  return (
    <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0 mt-0.5">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8" />
        <path d="m21 21-4.35-4.35" />
      </svg>
    </div>
  );
}

function UserBubble({ message }: { message: UserMessage }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[75%] bg-blue-600 text-white text-sm px-4 py-3 rounded-2xl rounded-tr-sm leading-relaxed">
        {message.text}
      </div>
    </div>
  );
}

function AssistantBubble({ message }: { message: AssistantMessage }) {
  const { results, status, error } = message;
  const isStreaming = status === "streaming";

  return (
    <div className="flex gap-3">
      <AssistantAvatar />
      <div className="flex-1 min-w-0 space-y-3">
        {status === "loading" && <TypingIndicator />}

        {(status === "streaming" || status === "done") && (
          <div className="space-y-3">
            <p className="text-xs text-gray-500">
              {status === "done"
                ? `Found ${results.length} relevant passage${results.length !== 1 ? "s" : ""}`
                : "Retrieving relevant passages..."}
            </p>
            <SourceCarousel results={results} isStreaming={isStreaming} />
          </div>
        )}

        {status === "no_knowledge" && (
          <div className="text-sm text-gray-600 bg-white border border-gray-200 rounded-2xl rounded-tl-sm px-4 py-3 leading-relaxed">
            We do not have sufficient knowledge in our database to answer that question.
          </div>
        )}

        {status === "error" && (
          <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-2xl rounded-tl-sm px-4 py-3">
            {error ?? "Something went wrong. Please try again."}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatMessage({ message }: { message: Message }) {
  if (message.role === "user") return <UserBubble message={message} />;
  return <AssistantBubble message={message} />;
}
