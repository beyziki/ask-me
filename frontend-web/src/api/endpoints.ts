import { apiClient } from "./client";
import type {
  AskResponse,
  CodeExplainResponse,
  DocumentGroup,
  DocumentOut,
  QuizOut,
  QuizSource,
  SourceRef,
  SummaryOut,
  SummaryStatus,
  User,
} from "./types";

export async function checkHealth(): Promise<{ status: string; app: string }> {
  const { data } = await apiClient.get("/health");
  return data;
}

export async function login(username: string, password: string): Promise<User> {
  const { data } = await apiClient.post("/users/login", { username, password });
  return data;
}

export async function register(
  username: string,
  password: string,
  preferred_language: string = "tr"
): Promise<User> {
  const { data } = await apiClient.post("/users/register", {
    username,
    password,
    preferred_language,
  });
  return data;
}

export async function listDocuments(): Promise<DocumentOut[]> {
  const { data } = await apiClient.get("/documents");
  return data;
}

export async function uploadDocument(file: File, groupId: number | null = null): Promise<DocumentOut> {
  const form = new FormData();
  form.append("file", file);
  if (groupId !== null) form.append("group_id", String(groupId));
  const { data } = await apiClient.post("/documents/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

/** Yanlışlıkla eklenen ya da artık istenmeyen bir dosyayı siler (chunk'ları ve
 * FAISS vektörleri de dahil, bkz. backend/app/api/documents.py:delete_document). */
export async function deleteDocument(documentId: number): Promise<void> {
  await apiClient.delete(`/documents/${documentId}`);
}

export async function listDocumentGroups(): Promise<DocumentGroup[]> {
  const { data } = await apiClient.get("/documents/groups");
  return data;
}

export async function createDocumentGroup(name: string): Promise<DocumentGroup> {
  const { data } = await apiClient.post<DocumentGroup>("/documents/groups", { name });
  return data;
}

export async function deleteDocumentGroup(groupId: number): Promise<void> {
  await apiClient.delete(`/documents/groups/${groupId}`);
}

/** Dokümanı bir gruba atar; `groupId: null` dokümanı grupsuz bırakır. */
export async function assignDocumentGroup(
  documentId: number,
  groupId: number | null
): Promise<DocumentOut> {
  const { data } = await apiClient.patch<DocumentOut>(`/documents/${documentId}/group`, {
    group_id: groupId,
  });
  return data;
}

export async function askQuestion(
  question: string,
  documentIds: number[] | null = null,
  language: string | null = null
): Promise<AskResponse> {
  // Foundry Local soğuk başlangıçta veya üretim sırasında uzun sürebiliyor
  // (bkz. backend/app/services/llm.py); varsayılan axios timeout'unu burada
  // bilinçli olarak kaldırıyoruz.
  const { data } = await apiClient.post<AskResponse>(
    "/ask",
    { question, document_ids: documentIds, language },
    { timeout: 0 }
  );
  return data;
}

/**
 * Bir SSE (Server-Sent Events) POST isteğini tüketip her olayı `onEvent`'e
 * iletir. `EventSource` yerine `fetch` + `ReadableStream` kullanıyoruz çünkü
 * tarayıcının yerleşik `EventSource` API'si yalnızca GET destekliyor; bizim
 * isteklerimiz ise (soru/doküman id'leri, quiz parametreleri gibi) bir JSON
 * gövdesi taşıyan POST. `/ask/stream` ve `/quiz/stream` ortak kullanıyor
 * (bkz. backend/app/api/ask.py:_sse ve backend/app/api/quiz.py:_sse) —
 * her ikisinin de olay formatı `data: <json>\n\n`, olay tipi JSON'daki
 * `type` alanıyla ayırt ediliyor.
 */
async function postSSE(
  path: string,
  body: unknown,
  userId: number,
  onEvent: (event: any) => void,
  signal?: AbortSignal
): Promise<void> {
  const baseURL = apiClient.defaults.baseURL ?? "http://127.0.0.1:8000";

  const res = await fetch(`${baseURL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Id": String(userId),
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `İstek başarısız (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE olayları "\n\n" ile ayrılır; tamponda henüz tamamlanmamış (yarım
    // gelen) bir olay varsa bir sonraki chunk'ı bekleriz.
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);

      const dataLine = rawEvent.split("\n").find((line) => line.startsWith("data:"));
      if (!dataLine) continue;
      const jsonText = dataLine.slice("data:".length).trim();
      if (!jsonText) continue;

      onEvent(JSON.parse(jsonText));
    }
  }
}

export interface AskStreamHandlers {
  /** `hasContext=false` ise hybrid RAG yeterince ilgili bir parça bulamadı:
   * cevap yüklenen dosyalardan değil, modelin genel bilgisinden geliyor
   * (bkz. backend/app/api/ask.py:_retrieve). */
  onSources?: (sources: SourceRef[], hasContext: boolean) => void;
  onToken?: (token: string) => void;
  onError?: (detail: string) => void;
  onDone?: () => void;
}

/** `/ask/stream` uç noktasını tüketir; olay tipleri: "sources" | "token" | "error" | "done". */
export async function askQuestionStream(
  params: {
    question: string;
    documentIds?: number[] | null;
    language?: string | null;
    userId: number;
  },
  handlers: AskStreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  await postSSE(
    "/ask/stream",
    {
      question: params.question,
      document_ids: params.documentIds ?? null,
      language: params.language ?? null,
    },
    params.userId,
    (event) => {
      switch (event.type) {
        case "sources":
          handlers.onSources?.(
            event.sources as SourceRef[],
            // Eski backend sürümleri bu alanı göndermiyordu; yokluğunda
            // eski davranışa (kaynak var) düşüyoruz.
            event.has_context !== false
          );
          break;
        case "token":
          handlers.onToken?.(event.content as string);
          break;
        case "error":
          handlers.onError?.(event.detail as string);
          break;
        case "done":
          handlers.onDone?.();
          break;
      }
    },
    signal
  );
}

export async function createQuiz(
  documentId: number,
  numQuestions: number = 5,
  source: QuizSource = "auto"
): Promise<QuizOut> {
  const { data } = await apiClient.post<QuizOut>(
    "/quiz",
    { document_id: documentId, num_questions: numQuestions, source },
    { timeout: 0 }
  );
  return data;
}

export interface QuizStreamHandlers {
  /** Üretim başlamadan önce, quiz'in hangi metinden üretildiğini bildirir —
   * arayüz "özetten üretiliyor" rozetini hemen gösterebilsin diye. */
  onSource?: (used: "summary" | "chunks") => void;
  /** Modelin ham (henüz JSON olarak ayrıştırılmamış) çıktı parçaları — canlı
   * bir ilerleme göstergesi için (bkz. QuizPage.tsx). */
  onToken?: (piece: string) => void;
  onError?: (detail: string) => void;
  /** Akış bitip sorular DB'ye kaydedildikten sonra, nihai quiz TEK SEFERDE gelir. */
  onResult?: (quiz: QuizOut) => void;
}

/** `/quiz/stream` uç noktasını tüketir; olay tipleri: "token" | "error" | "result". */
export async function createQuizStream(
  params: {
    documentId: number;
    numQuestions: number;
    userId: number;
    source?: QuizSource;
  },
  handlers: QuizStreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  await postSSE(
    "/quiz/stream",
    {
      document_id: params.documentId,
      num_questions: params.numQuestions,
      source: params.source ?? "auto",
    },
    params.userId,
    (event) => {
      switch (event.type) {
        case "source":
          handlers.onSource?.(event.used as "summary" | "chunks");
          break;
        case "token":
          handlers.onToken?.(event.content as string);
          break;
        case "error":
          handlers.onError?.(event.detail as string);
          break;
        case "result":
          handlers.onResult?.(event.quiz as QuizOut);
          break;
      }
    },
    signal
  );
}

export async function explainCode(documentId: number): Promise<CodeExplainResponse> {
  const { data } = await apiClient.post<CodeExplainResponse>(
    "/code/explain",
    { document_id: documentId },
    { timeout: 0 }
  );
  return data;
}


// --- Özet -----------------------------------------------------------------

/** Kullanıcının hangi dokümanlarının özeti olduğunu tek istekte döner. */
export async function listSummaryStatus(): Promise<SummaryStatus[]> {
  const { data } = await apiClient.get<SummaryStatus[]>("/summary");
  return data;
}

/** Kayıtlı özeti getirir; henüz üretilmemişse 404 atar. */
export async function getSummary(documentId: number): Promise<SummaryOut> {
  const { data } = await apiClient.get<SummaryOut>(`/summary/${documentId}`);
  return data;
}

/** Kayıtlı özeti siler (yeniden ürettirmek için). */
export async function deleteSummary(documentId: number): Promise<void> {
  await apiClient.delete(`/summary/${documentId}`);
}

export interface SummaryStreamHandlers {
  /** Uzun dokümanlarda, nihai özet akmaya başlamadan önceki map adımının
   * ilerlemesi (ör. "3/7"). Kısa dokümanlarda hiç gelmez. */
  onProgress?: (detail: string) => void;
  onToken?: (token: string) => void;
  onError?: (detail: string) => void;
  onDone?: () => void;
}

/** `/summary/{id}/stream` uç noktasını tüketir; özet bitince backend onu
 * otomatik olarak kaydeder. */
export async function createSummaryStream(
  params: { documentId: number; userId: number },
  handlers: SummaryStreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  await postSSE(
    `/summary/${params.documentId}/stream`,
    {},
    params.userId,
    (event) => {
      switch (event.type) {
        case "progress":
          handlers.onProgress?.(event.detail as string);
          break;
        case "token":
          handlers.onToken?.(event.content as string);
          break;
        case "error":
          handlers.onError?.(event.detail as string);
          break;
        case "done":
          handlers.onDone?.();
          break;
      }
    },
    signal
  );
}
