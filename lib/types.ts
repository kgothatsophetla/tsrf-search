export interface SearchResult {
  index: number;
  text: string;
  page: number;
  source: string;
  score: number;
  question?: string;  // present when result came from a Q&A corpus entry
}

export interface UserMessage {
  id: string;
  role: "user";
  text: string;
}

export interface AssistantMessage {
  id: string;
  role: "assistant";
  answer: string;
  results: SearchResult[];
  status: "loading" | "streaming" | "done" | "no_knowledge" | "error";
  error?: string;
}

export type Message = UserMessage | AssistantMessage;

export interface HistoryEntry {
  question: string;           // what the user typed
  matched_question?: string;  // top corpus question that matched (if any)
  answer?: string;            // assistant's reply (for Claude conversation context)
}
