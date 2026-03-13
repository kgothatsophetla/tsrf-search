"use client";

import { FormEvent, useState } from "react";

interface Props {
  onSearch: (query: string) => void;
  disabled: boolean;
}

export default function SearchBar({ onSearch, disabled }: Props) {
  const [query, setQuery] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = query.trim();
    if (trimmed) {
      onSearch(trimmed);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="flex gap-3">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about TSRF..."
          disabled={disabled}
          autoFocus
          className="
            flex-1 px-4 py-3 text-sm border border-gray-300 rounded-lg
            focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
            disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed
            transition-shadow
          "
        />
        <button
          type="submit"
          disabled={disabled || !query.trim()}
          className="
            px-5 py-3 bg-blue-600 text-white text-sm font-medium rounded-lg
            hover:bg-blue-700 active:bg-blue-800
            focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
            disabled:opacity-50 disabled:cursor-not-allowed
            transition-colors
          "
        >
          Search
        </button>
      </div>
    </form>
  );
}
