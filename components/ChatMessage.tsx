"use client";

import { useState } from "react";
import type { AssistantMessage, Message, SearchResult, UserMessage } from "@/lib/types";

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 h-6">
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

function AssistantAvatar() {
  return (
    <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0 mt-1">
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

function AnswerText({ text }: { text: string }) {
  // Render plain text preserving newlines
  return (
    <div className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap">
      {text}
    </div>
  );
}

function SourcesToggle({ results }: { results: SearchResult[] }) {
  const [open, setOpen] = useState(false);

  if (results.length === 0) return null;

  return (
    <div className="mt-3 pt-3 border-t border-gray-100">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-600 transition-colors"
      >
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className={`transition-transform ${open ? "rotate-90" : ""}`}
        >
          <path d="M9 18l6-6-6-6" />
        </svg>
        {results.length} source{results.length !== 1 ? "s" : ""}
      </button>

      {open && (
        <ul className="mt-2 space-y-1.5">
          {results.map((r, i) => (
            <li key={i} className="flex items-start gap-2 text-xs text-gray-500">
              <span className="mt-0.5 text-gray-300">•</span>
              <span>
                {r.question ? (
                  <span className="text-blue-500">{r.question}</span>
                ) : (
                  r.text.slice(0, 80) + (r.text.length > 80 ? "…" : "")
                )}
                <span className="ml-1.5 text-gray-400">— p.{r.page}</span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function AssistantMessage_({ message }: { message: AssistantMessage }) {
  const { answer, results, status, error } = message;

  return (
    <div className="flex gap-3">
      <AssistantAvatar />
      <div className="flex-1 min-w-0 py-1">
        {status === "loading" && <TypingIndicator />}

        {(status === "streaming" || status === "done") && (
          <>
            <AnswerText text={answer} />
            <SourcesToggle results={results} />
          </>
        )}

        {status === "no_knowledge" && (
          <p className="text-sm text-gray-500 leading-relaxed">
            Sorry we are unable to assist with your enquiry. Please contact the Fund Call Centre on 087 405 6377 or visit one of the Fund Walk-in Centres. Go to our Contact page to see the details of the various Walk-in Centres.
          </p>
        )}

        {status === "error" && (
          <p className="text-sm text-red-500 leading-relaxed">
            {error ?? "Something went wrong. Please try again."}
          </p>
        )}
      </div>
    </div>
  );
}

export default function ChatMessage({ message }: { message: Message }) {
  if (message.role === "user") return <UserBubble message={message} />;
  return <AssistantMessage_ message={message} />;
}
