import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { CheckCircle2, XCircle, RotateCcw, Sparkles } from "lucide-react";
import { createQuizStream, listSummaryStatus } from "../api/endpoints";
import type { QuizOut, QuizSource } from "../api/types";
import { Button, Card, Spinner } from "../components/ui";
import HistoryPanel from "../components/HistoryPanel";
import { useAuth } from "../context/AuthContext";
import { useDocuments } from "../context/DocumentsContext";

interface QuizEntry {
  id: string;
  documentId: number;
  documentName: string;
  label: string;
  createdAt: number;
  quiz: QuizOut;
  answers: Record<number, string>;
  revealed: Record<number, boolean>;
}

const HISTORY_KEY = "ask-me:quiz-history";

function loadHistory(): QuizEntry[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? (JSON.parse(raw) as QuizEntry[]) : [];
  } catch {
    return [];
  }
}

function saveHistory(entries: QuizEntry[]) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(entries));
}

export default function QuizPage({
  historyPanelSide = "right",
}: {
  /** Split View'da (bkz. SplitViewPage.tsx) iki geçmiş paneli çakışmasın
   * diye; tek başına (route: /quiz) her zaman varsayılan "right". */
  historyPanelSide?: "left" | "right";
} = {}) {
  const { user } = useAuth();
  const { documents, groups } = useDocuments();
  const [documentId, setDocumentId] = useState<number | null>(null);
  // Doküman listesini daraltmak için isteğe bağlı grup filtresi; quiz yine
  // TEK bir dokümandan üretiliyor (bkz. handleGenerate), bu sadece çok
  // dosya olduğunda doğru dokümanı bulmayı kolaylaştırıyor. null = tüm dosyalar.
  const [groupFilter, setGroupFilter] = useState<number | null>(null);
  const filteredDocuments =
    groupFilter === null ? documents : documents.filter((d) => d.group_id === groupFilter);
  const [numQuestions, setNumQuestions] = useState(5);
  // Quiz hangi metinden üretilsin? "auto" = özet varsa ondan (belirgin
  // biçimde daha hızlı, çünkü özet zaten damıtılmış ve kısa bir metin),
  // yoksa ham doküman parçalarından. Bkz. backend/app/api/quiz.py.
  const [quizSource, setQuizSource] = useState<QuizSource>("auto");
  // Hangi dokümanların özeti var — "özetten üret" seçeneğini aktif
  // edip etmeyeceğimize ve bilgilendirme metnine karar vermek için.
  const [withSummary, setWithSummary] = useState<Set<number>>(new Set());
  // Üretim sırasında backend'in bildirdiği gerçek kaynak (bkz.
  // api/endpoints.ts:QuizStreamHandlers.onSource).
  const [usedSource, setUsedSource] = useState<"summary" | "chunks" | null>(null);
  const [loading, setLoading] = useState(false);
  // Quiz çıktısı ham JSON olduğu için (bkz. backend/app/api/quiz.py:create_quiz_stream)
  // token'ları kullanıcıya olduğu gibi göstermek yerine, sadece akışın canlı
  // olduğunu belli eden bir sayaç (o ana kadar alınan karakter sayısı)
  // tutuyoruz — çıplak JSON parçaları göstermek kafa karıştırıcı olurdu.
  const [streamedChars, setStreamedChars] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Bu ekrandan çıkıp geri döndüğünde (ör. Soru Sor'a geçip tekrar Quiz'e
  // gelince) sadece component state'i kullansaydık üretilen sorular kaybolurdu
  // (component her mount'ta sıfırdan başlar). Bunun önüne geçmek için tüm quiz
  // geçmişini localStorage'da tutuyoruz; ekrana her girişte oradan yükleniyor.
  const [history, setHistory] = useState<QuizEntry[]>(() => loadHistory());
  const [activeId, setActiveId] = useState<string | null>(() => loadHistory()[0]?.id ?? null);

  const active = history.find((h) => h.id === activeId) ?? null;

  // Özet ekranındaki "Bu özetten quiz üret" butonu buraya bir documentId ile
  // yönlendiriyor (bkz. SummaryPage.tsx); o dokümanı otomatik seçiyoruz.
  const location = useLocation();
  const incomingDocumentId = (location.state as { documentId?: number } | null)?.documentId;
  useEffect(() => {
    if (typeof incomingDocumentId === "number") setDocumentId(incomingDocumentId);
  }, [incomingDocumentId]);

  // Hangi dokümanların özeti olduğunu bir kez çekiyoruz; "özetten üret"
  // seçeneğini yalnızca gerçekten özeti olan dokümanlarda etkin kılmak için.
  useEffect(() => {
    let cancelled = false;
    listSummaryStatus()
      .then((rows) => {
        if (cancelled) return;
        setWithSummary(new Set(rows.filter((r) => r.has_summary).map((r) => r.document_id)));
      })
      .catch(() => {
        // Yalnızca bir ipucu; alınamazsa "auto" zaten doğru davranıyor.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedHasSummary = documentId !== null && withSummary.has(documentId);

  // Doküman listesi artık DocumentsContext'ten (önbellekli) geliyor; burada
  // henüz bir seçim yapılmamışsa ya da grup filtresi mevcut seçimi listeden
  // düşürdüyse (filtrelenmiş listede yoksa) ilk dokümanı varsayılan seçiyoruz.
  useEffect(() => {
    if (filteredDocuments.length === 0) {
      if (documentId !== null) setDocumentId(null);
      return;
    }
    if (documentId === null || !filteredDocuments.some((d) => d.id === documentId)) {
      setDocumentId(filteredDocuments[0].id);
    }
  }, [filteredDocuments, documentId]);

  function persist(entries: QuizEntry[]) {
    setHistory(entries);
    saveHistory(entries);
  }

  function updateActive(patch: Partial<QuizEntry>) {
    if (!activeId) return;
    persist(history.map((h) => (h.id === activeId ? { ...h, ...patch } : h)));
  }

  async function handleGenerate() {
    if (!documentId || !user) return;
    // `documentId` yukarıdaki guard'la `number`'a daraltıldı, ama bu daralma
    // aşağıdaki iç içe fonksiyonun (`addQuizToHistory`) gövdesine taşınmıyor
    // (TypeScript, closure'ların ileride farklı bir anda çalışabileceğini
    // varsayıp orada tekrar geniş tipe dönüyor); bu yüzden `docId` adıyla
    // düz bir sabite kopyalıyoruz.
    const docId = documentId;
    const doc = documents.find((d) => d.id === docId);
    setLoading(true);
    setError(null);
    setStreamedChars(0);
    setUsedSource(null);

    function addQuizToHistory(quiz: QuizOut) {
      const countForDoc = history.filter((h) => h.documentId === docId).length;
      const entry: QuizEntry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        documentId: docId,
        documentName: doc?.filename ?? "Doküman",
        label: `${doc?.filename ?? "Doküman"} · Quiz ${countForDoc + 1}`,
        createdAt: Date.now(),
        quiz,
        answers: {},
        revealed: {},
      };
      const next = [entry, ...history];
      persist(next);
      setActiveId(entry.id);
    }

    try {
      await createQuizStream(
        { documentId: docId, numQuestions, userId: user.id, source: quizSource },
        {
          onSource: (used) => setUsedSource(used),
          onToken: (piece) => setStreamedChars((c) => c + piece.length),
          onError: (detail) => setError(detail),
          onResult: addQuizToHistory,
        }
      );
    } catch (err) {
      setError(err instanceof Error && err.message ? err.message : "Quiz üretilemedi.");
    } finally {
      setLoading(false);
    }
  }

  function removeEntry(id: string) {
    const next = history.filter((h) => h.id !== id);
    persist(next);
    if (activeId === id) setActiveId(next[0]?.id ?? null);
  }

  function reveal(index: number) {
    if (!active) return;
    updateActive({ revealed: { ...active.revealed, [index]: true } });
  }

  function setAnswer(index: number, value: string) {
    if (!active) return;
    updateActive({ answers: { ...active.answers, [index]: value } });
  }

  const correctCount = active
    ? active.quiz.questions.filter(
        (q, i) =>
          active.revealed[i] &&
          (active.answers[i] ?? "").trim().toLowerCase() === q.answer.trim().toLowerCase()
      ).length
    : 0;
  const revealedCount = active ? Object.values(active.revealed).filter(Boolean).length : 0;

  return (
    <div className="mx-auto max-w-3xl space-y-8">
        <div>
          <h1 className="text-2xl font-semibold text-white">Quiz</h1>
          <p className="mt-1 text-sm text-zinc-500">
            Bir dokümandan otomatik çalışma soruları üret ve kendini test et.
          </p>
        </div>

        <Card className="space-y-4 p-5">
          {documents.length === 0 ? (
            <p className="text-sm text-zinc-500">Önce Dosya Yükle ekranından bir kaynak yükle.</p>
          ) : (
            <>
              {groups.length > 0 && (
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-500">Grup (isteğe bağlı filtre)</label>
                  <select
                    value={groupFilter ?? ""}
                    onChange={(e) => setGroupFilter(e.target.value ? Number(e.target.value) : null)}
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-950/60 px-3 py-2.5 text-sm text-zinc-100 outline-none focus:border-indigo-500"
                  >
                    <option value="">Tüm dosyalar</option>
                    {groups.map((g) => (
                      <option key={g.id} value={g.id}>
                        {g.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {filteredDocuments.length === 0 ? (
                <p className="text-sm text-zinc-500">Bu grupta dosya yok.</p>
              ) : (
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-zinc-500">Doküman</label>
                    <select
                      value={documentId ?? ""}
                      onChange={(e) => setDocumentId(Number(e.target.value))}
                      className="w-full rounded-lg border border-zinc-700 bg-zinc-950/60 px-3 py-2.5 text-sm text-zinc-100 outline-none focus:border-indigo-500"
                    >
                      {filteredDocuments.map((doc) => (
                        <option key={doc.id} value={doc.id}>
                          {doc.filename}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-zinc-500">Soru sayısı</label>
                    <input
                      type="number"
                      min={1}
                      max={15}
                      value={numQuestions}
                      onChange={(e) => setNumQuestions(Number(e.target.value))}
                      className="w-full rounded-lg border border-zinc-700 bg-zinc-950/60 px-3 py-2.5 text-sm text-zinc-100 outline-none focus:border-indigo-500"
                    />
                  </div>
                </div>
              )}

              {filteredDocuments.length > 0 && (
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-500">Kaynak</label>
                  <div className="flex flex-wrap gap-1.5">
                    {(
                      [
                        { value: "auto", label: "Otomatik" },
                        { value: "summary", label: "Özetten" },
                        { value: "chunks", label: "Ham metinden" },
                      ] as { value: QuizSource; label: string }[]
                    ).map(({ value, label }) => {
                      // Özeti olmayan bir dokümanda "Özetten" seçilemez.
                      const disabled = value === "summary" && !selectedHasSummary;
                      const selected = quizSource === value;
                      return (
                        <button
                          key={value}
                          type="button"
                          disabled={disabled}
                          onClick={() => setQuizSource(value)}
                          title={
                            disabled
                              ? "Bu doküman için henüz özet yok — önce Özet ekranından çıkar"
                              : undefined
                          }
                          className={`rounded-full border px-3 py-1 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-40 ${
                            selected
                              ? "border-indigo-500 bg-indigo-500/15 text-indigo-300"
                              : "border-zinc-800 text-zinc-400 hover:border-zinc-700"
                          }`}
                        >
                          {label}
                        </button>
                      );
                    })}
                  </div>
                  <p className="text-[11px] text-zinc-600">
                    {quizSource === "chunks"
                      ? "Doküman parçalarından üretilir — daha yavaş, kapsam örneklemeyle sınırlı."
                      : selectedHasSummary
                        ? "Özetten üretilir — çok daha hızlı ve dokümanın tamamını kapsar."
                        : "Bu dokümanın özeti yok; ham metinden üretilecek. Özet çıkarırsan quiz hızlanır."}
                  </p>
                </div>
              )}

              <Button onClick={handleGenerate} disabled={loading || !documentId} className="w-full sm:w-auto">
                {loading ? (
                  <>
                    <Spinner className="h-4 w-4" /> Üretiliyor...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" strokeWidth={1.75} /> Quiz Oluştur
                  </>
                )}
              </Button>

              {loading && usedSource && (
                <p className="text-xs text-zinc-500">
                  {usedSource === "summary"
                    ? "Özetten üretiliyor (hızlı yol)"
                    : "Ham doküman parçalarından üretiliyor"}
                </p>
              )}

              {loading && (
                // Quiz üretimi (ham JSON, tek nihai sonuç olarak gelir; bkz.
                // api/endpoints.ts:createQuizStream) toplamda hâlâ onlarca
                // saniye/birkaç dakika sürebiliyor. Sabit bir spinner yerine,
                // model çıktısı geldikçe büyüyen bir sayaç göstererek
                // "donmadı, devam ediyor" hissini veriyoruz.
                <p className="text-xs text-zinc-500">
                  {streamedChars > 0
                    ? `Sorular yazılıyor... (${streamedChars} karakter alındı)`
                    : "Kaynak taranıyor, model başlatılıyor..."}
                </p>
              )}
            </>
          )}

          {error && (
            <div className="rounded-lg border border-red-900/60 bg-red-950/30 px-4 py-3 text-sm text-red-300">
              {error}
            </div>
          )}
        </Card>

        {active && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-medium text-zinc-300">{active.label}</h2>
              {revealedCount > 0 && (
                <span className="text-xs text-zinc-500">
                  Skor: <span className="font-medium text-zinc-300">{correctCount}</span> /{" "}
                  {revealedCount}
                </span>
              )}
            </div>

            {active.quiz.questions.map((q, i) => {
              const isRevealed = !!active.revealed[i];
              const userAnswer = active.answers[i] ?? "";
              const isCorrect = userAnswer.trim().toLowerCase() === q.answer.trim().toLowerCase();

              return (
                <Card key={i} className="space-y-3 p-5">
                  <p className="text-sm font-medium text-zinc-200">
                    {i + 1}. {q.question}
                  </p>

                  {q.options ? (
                    <div className="grid gap-2 sm:grid-cols-2">
                      {q.options.map((opt) => {
                        const selected = userAnswer === opt;
                        const showCorrect =
                          isRevealed && opt.trim().toLowerCase() === q.answer.trim().toLowerCase();
                        const showWrong = isRevealed && selected && !showCorrect;
                        return (
                          <button
                            key={opt}
                            type="button"
                            disabled={isRevealed}
                            onClick={() => setAnswer(i, opt)}
                            className={`flex items-center justify-between rounded-lg border px-3 py-2.5 text-left text-sm transition ${
                              showCorrect
                                ? "border-emerald-600/60 bg-emerald-950/30 text-emerald-300"
                                : showWrong
                                  ? "border-red-900/60 bg-red-950/30 text-red-300"
                                  : selected
                                    ? "border-indigo-500 bg-indigo-500/10 text-indigo-200"
                                    : "border-zinc-800 text-zinc-300 hover:border-zinc-700"
                            } disabled:cursor-default`}
                          >
                            {opt}
                            {showCorrect && <CheckCircle2 className="h-4 w-4" strokeWidth={1.75} />}
                            {showWrong && <XCircle className="h-4 w-4" strokeWidth={1.75} />}
                          </button>
                        );
                      })}
                    </div>
                  ) : (
                    <input
                      value={userAnswer}
                      disabled={isRevealed}
                      onChange={(e) => setAnswer(i, e.target.value)}
                      placeholder="Cevabını yaz..."
                      className="w-full rounded-lg border border-zinc-700 bg-zinc-950/60 px-3 py-2.5 text-sm text-zinc-100 outline-none focus:border-indigo-500 disabled:opacity-70"
                    />
                  )}

                  {!isRevealed ? (
                    <button
                      type="button"
                      onClick={() => reveal(i)}
                      disabled={!userAnswer}
                      className="text-xs font-medium text-indigo-400 hover:text-indigo-300 disabled:opacity-40"
                    >
                      Cevabı kontrol et
                    </button>
                  ) : (
                    <div
                      className={`flex items-center gap-1.5 text-xs font-medium ${
                        isCorrect ? "text-emerald-400" : "text-red-400"
                      }`}
                    >
                      {isCorrect ? (
                        <CheckCircle2 className="h-3.5 w-3.5" strokeWidth={1.75} />
                      ) : (
                        <XCircle className="h-3.5 w-3.5" strokeWidth={1.75} />
                      )}
                      {isCorrect ? "Doğru!" : `Doğru cevap: ${q.answer}`}
                    </div>
                  )}
                </Card>
              );
            })}

            <button
              type="button"
              onClick={handleGenerate}
              className="flex items-center gap-1.5 text-xs font-medium text-zinc-500 hover:text-zinc-300"
            >
              <RotateCcw className="h-3.5 w-3.5" strokeWidth={1.75} /> Yeni quiz üret
            </button>
          </div>
        )}

      <HistoryPanel
        title="Quiz Geçmişi"
        items={history.map((h) => ({
          id: h.id,
          label: h.label,
          meta: `${h.quiz.questions.length} soru`,
        }))}
        activeId={activeId}
        onSelect={setActiveId}
        onRemove={removeEntry}
        emptyText="Ürettiğin quizler burada listelenecek, ekranlar arası geçişte kaybolmayacaklar."
        side={historyPanelSide}
      />
    </div>
  );
}
