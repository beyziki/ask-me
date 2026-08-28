import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { Send, Sparkles, FileStack, Check, ChevronDown, Search, X } from "lucide-react";
import { askQuestionStream } from "../api/endpoints";
import type { DocumentOut, SourceRef } from "../api/types";
import { Card, Spinner } from "../components/ui";
import HistoryPanel from "../components/HistoryPanel";
import Markdown from "../components/Markdown";
import { useAuth } from "../context/AuthContext";
import { useDocuments } from "../context/DocumentsContext";

const UNGROUPED_KEY = "ungrouped";

function formatShortDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString("tr-TR", { day: "2-digit", month: "2-digit", year: "2-digit" });
  } catch {
    return iso;
  }
}

/**
 * Soru sorulurken hangi dosya(lar)ın kaynak olarak kullanılacağını seçmek
 * için Kart. Seçim TAMAMEN İSTEĞE BAĞLI: hiçbir şey seçilmezse (varsayılan)
 * backend'e `document_ids: null` gönderilir ve hybrid RAG kullanıcının TÜM
 * dosyalarında arama yapar (bkz. backend/app/api/ask.py:_retrieve) — yani
 * dosyalarda olmayan/genel bir soru sormak için önce bir dosya seçmeye
 * zorlanmıyorsun. Belirli dosya(lar) seçersen arama yalnızca onlarla
 * sınırlanır (daha hızlı ve daha isabetli olabilir).
 *
 * Onlarca dosya olduğunda düz bir "chip duvarı" hem kalabalık hem de aynı
 * isimle başlayan (kesilmiş) dosyaları ayırt etmeyi zorlaştırıyordu (bkz.
 * kullanıcı ekran görüntüsü). Bunun yerine: bir arama kutusu, grupları
 * daraltılıp genişletilebilen (accordion) bölümler haline getiriyoruz, ve
 * her satırda TAM dosya adını + yükleme tarihini gösteriyoruz (aynı isimli
 * dosyaları tarihe bakarak ayırt edebilesin diye).
 */
function DocumentPicker({
  documents,
  groups,
  selectedIds,
  onToggleDocument,
  onToggleGroup,
  onClear,
}: {
  documents: DocumentOut[];
  groups: { id: number; name: string }[];
  selectedIds: number[];
  onToggleDocument: (id: number) => void;
  onToggleGroup: (docIds: number[]) => void;
  onClear: () => void;
}) {
  const [search, setSearch] = useState("");
  // Hangi grup bölümlerinin elle açıldığı — varsayılan olarak hepsi kapalı
  // (accordion); arama yaparken eşleşen bölümler bu sete bakılmaksızın
  // otomatik açılıyor (bkz. isOpen).
  const [openKeys, setOpenKeys] = useState<Set<string>>(new Set());

  const buckets = useMemo(() => {
    const byGroup = new Map<number | typeof UNGROUPED_KEY, DocumentOut[]>();
    for (const doc of documents) {
      const key = doc.group_id ?? UNGROUPED_KEY;
      if (!byGroup.has(key)) byGroup.set(key, []);
      byGroup.get(key)!.push(doc);
    }
    const ordered: { key: number | typeof UNGROUPED_KEY; label: string; docs: DocumentOut[] }[] = [];
    for (const group of groups) {
      const docs = byGroup.get(group.id);
      if (docs && docs.length > 0) ordered.push({ key: group.id, label: group.name, docs });
    }
    const ungrouped = byGroup.get(UNGROUPED_KEY);
    if (ungrouped && ungrouped.length > 0) {
      ordered.push({ key: UNGROUPED_KEY, label: "Grupsuz", docs: ungrouped });
    }
    return ordered;
  }, [documents, groups]);

  const query = search.trim().toLowerCase();
  const visibleBuckets = useMemo(() => {
    if (!query) return buckets;
    return buckets
      .map((bucket) => ({
        ...bucket,
        docs: bucket.docs.filter((d) => d.filename.toLowerCase().includes(query)),
      }))
      .filter((bucket) => bucket.docs.length > 0);
  }, [buckets, query]);

  function toggleOpen(key: string) {
    setOpenKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  const allSelected = selectedIds.length === 0;

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <button
          type="button"
          onClick={onClear}
          className={`rounded-full border px-3 py-1 text-xs font-medium transition ${
            allSelected
              ? "border-indigo-500 bg-indigo-500/15 text-indigo-300"
              : "border-zinc-800 text-zinc-400 hover:border-zinc-700"
          }`}
        >
          Tüm dosyalar
        </button>
        {selectedIds.length > 0 && (
          <span className="text-xs text-zinc-600">{selectedIds.length} dosya seçili</span>
        )}
      </div>

      {documents.length > 5 && (
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-600" strokeWidth={1.75} />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Dosya ara..."
            className="w-full rounded-lg border border-zinc-800 bg-zinc-950/60 py-1.5 pl-8 pr-7 text-xs text-zinc-200 placeholder:text-zinc-600 outline-none focus:border-indigo-500"
          />
          {search && (
            <button
              type="button"
              onClick={() => setSearch("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-600 hover:text-zinc-300"
            >
              <X className="h-3.5 w-3.5" strokeWidth={2} />
            </button>
          )}
        </div>
      )}

      {query && visibleBuckets.length === 0 && (
        <p className="px-1 text-xs text-zinc-600">"{search}" ile eşleşen dosya yok.</p>
      )}

      <div className="space-y-1.5">
        {visibleBuckets.map((bucket) => {
          const key = String(bucket.key);
          const open = query !== "" || openKeys.has(key);
          const groupSelected =
            bucket.docs.length > 0 && bucket.docs.every((d) => selectedIds.includes(d.id));
          const someSelected = !groupSelected && bucket.docs.some((d) => selectedIds.includes(d.id));

          return (
            <div key={key} className="overflow-hidden rounded-lg border border-zinc-800">
              <div className="flex items-center gap-1 bg-zinc-900/60 pr-1.5">
                <button
                  type="button"
                  onClick={() => toggleOpen(key)}
                  className="flex flex-1 items-center gap-1.5 px-2.5 py-1.5 text-left text-xs font-medium text-zinc-300 hover:text-zinc-100"
                >
                  <ChevronDown
                    className={`h-3.5 w-3.5 shrink-0 text-zinc-600 transition-transform ${open ? "rotate-0" : "-rotate-90"}`}
                    strokeWidth={2}
                  />
                  {bucket.label}
                  <span className="text-zinc-600">· {bucket.docs.length}</span>
                </button>
                <button
                  type="button"
                  onClick={() => onToggleGroup(bucket.docs.map((d) => d.id))}
                  title="Bu gruptaki tüm dosyaları seç/kaldır"
                  className={`rounded-md border px-2 py-1 text-[11px] font-medium transition ${
                    groupSelected
                      ? "border-indigo-500 bg-indigo-500/15 text-indigo-300"
                      : someSelected
                        ? "border-indigo-800 text-indigo-300"
                        : "border-zinc-800 text-zinc-500 hover:border-zinc-700"
                  }`}
                >
                  {groupSelected ? "Kaldır" : "Tümünü seç"}
                </button>
              </div>

              {open && (
                <div className="divide-y divide-zinc-800/80">
                  {bucket.docs.map((doc) => {
                    const selected = selectedIds.includes(doc.id);
                    return (
                      <button
                        key={doc.id}
                        type="button"
                        onClick={() => onToggleDocument(doc.id)}
                        title={doc.filename}
                        className={`flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-xs transition ${
                          selected ? "bg-indigo-500/10 text-indigo-200" : "text-zinc-400 hover:bg-zinc-900"
                        }`}
                      >
                        <span
                          className={`flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-[4px] border ${
                            selected ? "border-indigo-400 bg-indigo-500/20" : "border-zinc-700"
                          }`}
                        >
                          {selected && <Check className="h-2.5 w-2.5" strokeWidth={3} />}
                        </span>
                        <span className="min-w-0 flex-1 truncate">{doc.filename}</span>
                        <span className="shrink-0 text-[11px] text-zinc-600">
                          {formatShortDate(doc.uploaded_at)}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceRef[];
  /** false ise hybrid RAG yeterince ilgili bir parça bulamadı: cevap
   * yüklenen dosyalardan değil, modelin genel bilgisinden geliyor
   * (bkz. backend/app/api/ask.py:_retrieve). Eski sohbet geçmişinde bu
   * alan yok; `undefined` eski davranış (kaynağa dayalı cevap) sayılıyor. */
  hasContext?: boolean;
}

interface ChatSession {
  id: string;
  label: string;
  createdAt: number;
  messages: ChatMessage[];
}

const HISTORY_KEY = "ask-me:chat-history";

// Backend artık /ask/stream ile cevabı token token akıtıyor (bkz.
// backend/app/api/ask.py:ask_stream ve api/endpoints.ts:askQuestionStream).
// Ama ilk token gelene kadar hâlâ görünür bir bekleme var: hybrid RAG araması
// + (soğuk başlangıçta) Foundry Local'in modeli belleğe yüklemesi onlarca
// saniye sürebiliyor. Bu ilk aşamada boş bir "yükleniyor" göstermek yerine
// adım adım durum mesajları gösteriyoruz; kaynaklar veya ilk token gelir
// gelmez bu yerini gerçek, canlı akan cevaba bırakıyor (aşağıdaki render'a bak).
const THINKING_STEPS = [
  "Kaynaklar taranıyor...",
  "İlgili parçalar sıralanıyor (hybrid RAG)...",
  "Model bağlamı işliyor...",
  "Cevap oluşturuluyor...",
];

function loadHistory(): ChatSession[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? (JSON.parse(raw) as ChatSession[]) : [];
  } catch {
    return [];
  }
}

function saveHistory(sessions: ChatSession[]) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(sessions));
}

function newId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/** Cevabın dosyalardan değil modelin genel bilgisinden geldiğini belirten
 * uyarı. Eskiden bu durum kullanıcıya HİÇ belli edilmiyordu: RAG alakasız
 * parçalar bulduğunda bile normal bir "kaynaklı cevap" gibi görünüyor,
 * kullanıcı da cevabın kendi notlarına dayandığını sanıyordu. */
function NoContextNotice() {
  return (
    <div className="rounded-lg border border-amber-900/60 bg-amber-950/30 px-3 py-2 text-xs text-amber-300/90">
      Dosyalarında bu soruyla ilgili bir bölüm bulunamadı — cevap modelin genel
      bilgisinden geliyor.
    </div>
  );
}

function SourcesList({ sources }: { sources: SourceRef[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="space-y-1.5 pl-1">
      <p className="text-xs font-medium text-zinc-600">Kaynaklar</p>
      {sources.map((s, i) => (
        <div
          key={i}
          className="rounded-lg border border-zinc-800/80 bg-zinc-900/40 px-3 py-2 text-xs text-zinc-500"
        >
          <span className="font-medium text-zinc-400">{s.filename}</span>
          {" · "}
          <span>parça #{s.chunk_index}</span>
          <p className="mt-1 truncate text-zinc-600">{s.snippet}</p>
        </div>
      ))}
    </div>
  );
}

export default function AskPage({
  historyPanelSide = "right",
}: {
  /** Split View'da (bkz. SplitViewPage.tsx) iki geçmiş paneli çakışmasın
   * diye; tek başına (route: /ask) her zaman varsayılan "right". */
  historyPanelSide?: "left" | "right";
} = {}) {
  const { user } = useAuth();
  const { documents, groups } = useDocuments();
  const [question, setQuestion] = useState("");
  const [thinking, setThinking] = useState(false);
  const [thinkingStep, setThinkingStep] = useState(0);
  // Hangi dosya(lar)ın kaynak olarak kullanılacağı — bkz. DocumentPicker'ın
  // üstündeki not. Boş dizi = "Tüm dosyalar" (varsayılan, seçim zorunlu değil).
  const [selectedDocIds, setSelectedDocIds] = useState<number[]>([]);

  function toggleDocument(id: number) {
    setSelectedDocIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  function toggleGroup(docIds: number[]) {
    setSelectedDocIds((prev) => {
      const allSelected = docIds.length > 0 && docIds.every((id) => prev.includes(id));
      if (allSelected) return prev.filter((id) => !docIds.includes(id));
      const merged = new Set(prev);
      docIds.forEach((id) => merged.add(id));
      return [...merged];
    });
  }
  // Akış sırasında canlı büyüyen cevap metni ve (LLM üretiminden önce gelen)
  // kaynaklar; bunlar sohbet geçmişine kalıcı olarak yalnızca akış bittiğinde
  // ("done" olayında) yazılır — her token'da localStorage'a yazmamak için
  // ayrı, kalıcı olmayan bir state olarak tutuluyor.
  const [streamingText, setStreamingText] = useState("");
  const [streamingSources, setStreamingSources] = useState<SourceRef[] | null>(null);
  // Akış sırasında gelen `sources` olayının `has_context` alanı (bkz.
  // api/endpoints.ts:AskStreamHandlers). Varsayılan true: aksi belli olana
  // kadar cevabın kaynaklara dayandığını varsayıyoruz.
  const [streamingHasContext, setStreamingHasContext] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Sohbet geçmişi de Quiz ekranındaki gibi localStorage'da tutuluyor;
  // başka bir ekrana geçip geri döndüğünde ya da sayfa yenilense bile
  // kaybolmuyor.
  const [sessions, setSessions] = useState<ChatSession[]>(() => loadHistory());
  const [activeId, setActiveId] = useState<string | null>(() => loadHistory()[0]?.id ?? null);

  const active = sessions.find((s) => s.id === activeId) ?? null;
  const messages = active?.messages ?? [];

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, thinking, streamingText]);

  useEffect(() => {
    if (!thinking) {
      setThinkingStep(0);
      return;
    }
    const interval = setInterval(() => {
      setThinkingStep((s) => Math.min(s + 1, THINKING_STEPS.length - 1));
    }, 2200);
    return () => clearInterval(interval);
  }, [thinking]);

  function persist(next: ChatSession[]) {
    setSessions(next);
    saveHistory(next);
  }

  function startNewChat() {
    setActiveId(null);
  }

  function removeSession(id: string) {
    const next = sessions.filter((s) => s.id !== id);
    persist(next);
    if (activeId === id) setActiveId(next[0]?.id ?? null);
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const q = question.trim();
    // `user` teorik olarak null olabilir (tip seviyesinde), ama bu sayfa
    // Layout tarafından yalnızca giriş yapılmışken render ediliyor (bkz.
    // components/Layout.tsx), bu yüzden burada olsa olsa savunmacı bir kontrol.
    if (!q || thinking || !user) return;

    let session = active;
    let all = sessions;
    if (!session) {
      session = {
        id: newId(),
        label: q.length > 48 ? `${q.slice(0, 48)}...` : q,
        createdAt: Date.now(),
        messages: [],
      };
      all = [session, ...sessions];
      setActiveId(session.id);
    }

    const userMsg: ChatMessage = { id: newId(), role: "user", content: q };
    let updatedSession: ChatSession = { ...session, messages: [...session.messages, userMsg] };
    persist(all.map((s) => (s.id === updatedSession.id ? updatedSession : s)));

    setQuestion("");
    setThinking(true);
    setError(null);
    setStreamingText("");
    setStreamingSources(null);
    setStreamingHasContext(true);

    // handleSubmit her istekte yeniden çağrıldığı için bu değişkenler yerel:
    // "done"/"error" anında en güncel akış içeriğine, React state güncellemelerinin
    // (asenkron) tamamlanmasını beklemeden doğrudan erişmek için tutuluyor.
    let liveText = "";
    let liveSources: SourceRef[] = [];
    let liveHasContext = true;

    function finalizeAssistantMessage() {
      const assistantMsg: ChatMessage = {
        id: newId(),
        role: "assistant",
        content: liveText,
        sources: liveSources,
        hasContext: liveHasContext,
      };
      updatedSession = { ...updatedSession, messages: [...updatedSession.messages, assistantMsg] };
      persist(
        (all.some((s) => s.id === updatedSession.id) ? all : [updatedSession, ...all]).map((s) =>
          s.id === updatedSession.id ? updatedSession : s
        )
      );
      setStreamingText("");
      setStreamingSources(null);
      setStreamingHasContext(true);
    }

    try {
      await askQuestionStream(
        {
          question: q,
          userId: user.id,
          documentIds: selectedDocIds.length > 0 ? selectedDocIds : null,
        },
        {
          onSources: (sources, hasContext) => {
            liveSources = sources;
            liveHasContext = hasContext;
            setStreamingSources(sources);
            setStreamingHasContext(hasContext);
          },
          onToken: (token) => {
            liveText += token;
            setStreamingText(liveText);
          },
          onError: (detail) => {
            // Kesintiye kadar bir metin akmışsa (ör. üretim ortasında Foundry
            // çöktü) onu kaybetmek yerine sohbete kaydedip hatayı ayrıca gösteriyoruz.
            if (liveText) finalizeAssistantMessage();
            setError(detail);
          },
          onDone: finalizeAssistantMessage,
        }
      );
    } catch (err) {
      if (liveText) finalizeAssistantMessage();
      setError(
        err instanceof Error && err.message
          ? err.message
          : "Cevap alınamadı. Foundry Local çalışıyor mu kontrol et."
      );
    } finally {
      setThinking(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-3xl flex-col" style={{ minHeight: "calc(100vh - 8rem)" }}>
        <div className="mb-4 space-y-3">
          <div>
            <h1 className="text-2xl font-semibold text-white">Soru Sor</h1>
            <p className="mt-1 flex items-center gap-1.5 text-sm text-zinc-500">
              <FileStack className="h-3.5 w-3.5" strokeWidth={1.75} />
              {documents.length === 0
                ? "Henüz yüklenmiş bir kaynak yok, önce Dosya Yükle ekranından ekle"
                : selectedDocIds.length === 0
                  ? `Tüm dosyalarda (${documents.length}) aranacak — dosyalarında olmayan bir şey soracaksan bir şey seçmene gerek yok`
                  : `Yalnızca seçili ${selectedDocIds.length} dosyada aranacak`}
            </p>
          </div>

          {documents.length > 0 && (
            <DocumentPicker
              documents={documents}
              groups={groups}
              selectedIds={selectedDocIds}
              onToggleDocument={toggleDocument}
              onToggleGroup={toggleGroup}
              onClear={() => setSelectedDocIds([])}
            />
          )}
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto pb-4">
          {messages.length === 0 && !thinking && (
            <Card className="p-10 text-center text-sm text-zinc-500">
              Kaynaklarınla ilgili bir soru sor, cevap ve kullandığı kaynak parçaları burada
              görünecek.
            </Card>
          )}

          {messages.map((m) =>
            m.role === "user" ? (
              <div key={m.id} className="flex justify-end">
                <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-indigo-600 px-4 py-2.5 text-sm text-white">
                  {m.content}
                </div>
              </div>
            ) : (
              <div key={m.id} className="flex justify-start">
                <div className="max-w-[85%] space-y-3">
                  <div className="rounded-2xl rounded-bl-sm border border-zinc-800 bg-zinc-900 px-4 py-3">
                    <Markdown>{m.content}</Markdown>
                  </div>
                  {m.hasContext === false && <NoContextNotice />}
                  {m.sources && <SourcesList sources={m.sources} />}
                </div>
              </div>
            )
          )}

          {thinking &&
            (streamingText || streamingSources ? (
              // İlk kaynaklar veya ilk token gelmiş: artık aşağıdaki
              // adım-adım "..." mesajı yerine gerçek, canlı akan cevabı
              // (ve kaynaklar geldiyse onları) gösteriyoruz.
              <div className="flex justify-start">
                <div className="max-w-[85%] space-y-3">
                  <div className="rounded-2xl rounded-bl-sm border border-zinc-800 bg-zinc-900 px-4 py-3">
                    <Markdown>{streamingText}</Markdown>
                    <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-indigo-400 align-middle" />
                  </div>
                  {!streamingHasContext && <NoContextNotice />}
                  {streamingSources && <SourcesList sources={streamingSources} />}
                </div>
              </div>
            ) : (
              <div className="flex justify-start">
                <div className="flex items-center gap-2 rounded-2xl rounded-bl-sm border border-zinc-800 bg-zinc-900 px-4 py-3 text-sm text-zinc-400">
                  <Sparkles className="h-4 w-4 animate-pulse text-indigo-400" strokeWidth={1.75} />
                  <span key={thinkingStep}>{THINKING_STEPS[thinkingStep]}</span>
                </div>
              </div>
            ))}

          {error && (
            <div className="rounded-lg border border-red-900/60 bg-red-950/30 px-4 py-3 text-sm text-red-300">
              {error}
            </div>
          )}

          <div ref={scrollRef} />
        </div>

        <form onSubmit={handleSubmit} className="sticky bottom-6 flex items-center gap-2">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Kaynaklarınla ilgili bir soru yaz..."
            disabled={thinking}
            className="w-full rounded-xl border border-zinc-700 bg-zinc-900 px-4 py-3 text-sm text-zinc-100 placeholder:text-zinc-500 outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={thinking || !question.trim()}
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-indigo-600 text-white transition hover:bg-indigo-500 disabled:opacity-50"
          >
            {thinking ? <Spinner className="h-4 w-4" /> : <Send className="h-4 w-4" strokeWidth={1.75} />}
          </button>
        </form>

      <HistoryPanel
        title="Sohbet Geçmişi"
        items={sessions.map((s) => ({ id: s.id, label: s.label, meta: `${s.messages.length} mesaj` }))}
        activeId={activeId}
        onSelect={setActiveId}
        onRemove={removeSession}
        onNew={startNewChat}
        emptyText="Sorduğun sorular burada listelenecek, ekranlar arası geçişte kaybolmayacaklar."
        side={historyPanelSide}
      />
    </div>
  );
}
