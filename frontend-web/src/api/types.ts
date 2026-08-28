// backend/app/models/schemas.py ile birebir eşleşen tipler.

export interface User {
  id: number;
  username: string;
  preferred_language: string;
}

export interface DocumentOut {
  id: number;
  filename: string;
  file_type: string;
  language: string | null;
  uploaded_at: string;
  group_id: number | null;
}

export interface DocumentGroup {
  id: number;
  name: string;
  created_at: string;
}

export interface SourceRef {
  document_id: number;
  filename: string;
  chunk_index: number;
  snippet: string;
}

export interface AskResponse {
  answer: string;
  sources: SourceRef[];
  /** Hybrid RAG yeterince ilgili bir parça bulabildi mi? False ise cevap
   * yüklenen dosyalardan değil, modelin genel bilgisinden geliyor. */
  has_context: boolean;
}

export interface QuizQuestion {
  question: string;
  options: string[] | null;
  answer: string;
}

export interface QuizOut {
  title: string;
  questions: QuizQuestion[];
  /** Quiz'in gerçekte hangi metinden üretildiği. */
  source: "summary" | "chunks";
}

export interface CodeExplainResponse {
  explanation: string;
}

export interface SummaryOut {
  document_id: number;
  filename: string;
  content: string;
  model_alias: string | null;
  updated_at: string;
}

export interface SummaryStatus {
  document_id: number;
  has_summary: boolean;
}

/** Quiz'in hangi metinden üretileceği. "auto": özet varsa ondan (daha hızlı),
 * yoksa ham parçalardan. */
export type QuizSource = "auto" | "summary" | "chunks";
