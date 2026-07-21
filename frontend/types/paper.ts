export type PaperAnalysis = {
  summary?: string;
  methodology?: string;
  method?: string;
  datasets?: string;
  dataset?: string;
  implementation?: string;
  contributions?: string;
  results?: string;
  limitations?: string;
  future_work?: string;
  equations?: string;
  research_gaps?: string;
  paper_info?: PaperInfo;
  debug_report?: DebugReport;
};

export type PaperInfo = {
  title?: string;
  authors?: string;
  year?: string;
  pages?: number;
};

export type DebugReport = {
  detected_sections?: string[];
  section_coverage?: number;
  missing_sections?: string[];
  unmatched_headings?: string[];
  section_traces?: Record<string, RetrievalTrace>;
  paper_info?: PaperInfo;
};

export type RetrievalTrace = {
  query?: string;
  retrieved_ids?: Array<string | number>;
  retrieved_chunk_ids?: Array<string | number>;
  retrieved_sections?: string[];
  retrieved_chunks?: Array<{
    chunk_id?: string | number;
    parent_id?: string | number;
    section?: string;
    pages?: Array<string | number>;
  }>;
  context_length?: number;
  faiss_scores?: Array<{ id?: string | number; score?: number; section?: string }>;
  bm25_scores?: Array<{ id?: string | number; score?: number; section?: string }>;
  rerank_scores?: Array<{ id?: string | number; score?: number; section?: string }>;
  final_context_length?: number;
  status?: string;
  fallback_mode?: string;
  chunks_sent_to_groq?: Array<Record<string, unknown>>;
};

export type ChatCitation = {
  section?: string;
  page?: string;
  chunk_id?: number | string | null;
  parent_id?: number | string | null;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  citations?: ChatCitation[];
};
