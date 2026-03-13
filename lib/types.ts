export interface SearchResult {
  index: number;
  text: string;
  page: number;
  source: string;
  score: number;
}

export interface UserMessage {
  id: string;
  role: "user";
  text: string;
}

export interface AssistantMessage {
  id: string;
  role: "assistant";
  results: SearchResult[];
  status: "loading" | "streaming" | "done" | "no_knowledge" | "error";
  error?: string;
}

export type Message = UserMessage | AssistantMessage;
