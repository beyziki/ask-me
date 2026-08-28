import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FileText, RefreshCw, Sparkles, Trash2, Brain } from "lucide-react";
import {
  createSummaryStream,
  deleteSummary,
  getSummary,
  listSummaryStatus,
} from "../api/endpoints";
import type { DocumentOut } from "../api/types";
import { Button, Card, Spinner } from "../components/ui";
import Markdown from "../components/Markdown";
import { useAuth } from "../context/AuthContext";
import { useDocuments } from "../context/DocumentsContext";

/**
 * Özet ekranı: bir dosyanın TAMAMINI özetler ve özeti kalıcı olarak saklar.
 *
 * Neden ayrı bir ekran (ve neden kalıcı): özet üretmek dokümanın tamamını
 * modelden geçirmeyi gerektiriyor — uzun dosyalarda parça parça (bkz.
 * backend/app/services/summary.py map-reduce). Bir kez üretip saklayınca hem
 * tekrar açıldığında anında geliyor, hem de quiz üretimi bu hazır ve kısa
 * metni kullanarak ham parçalardan üretmeye göre belirgin biçimde
 * hızlanıyor (bkz. backend/app/api/quiz.py:_load_quiz_context).
 */
export default function SummaryPage() {
  const { user } = useAuth();
  const { documents, loading: documentsLoading } = useDocuments();
  const navigate = useNavigate();

  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [summary, setSummary] = useState<string>("");
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const [modelAlias, setModelAlias] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Hangi dosyaların özeti var — dosya listesinde işaret göstermek için.
  const [withSummary, setWithSummary] = useState<Set<number>>(new Set());

  const refreshStatus = useCallback(async () => {
    try {
      const rows = await listSummaryStatus();
      setWithSummary(new Set(rows.filter((r) => r.has_summary).map((r) => r.document_id)));
    } catch {
      // Durum bilgisi yalnızca görsel bir ipucu; alınamazsa sayfayı bloke etme.
    }
  }, []);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  // Dosya seçilince varsa kayıtlı özeti getir (yoksa boş ekranla "Özet çıkar"
  // butonunu göster). Yeni bir üretim başlatmıyoruz — kullanıcı istemeden
  // pahalı bir işlem tetiklenmemeli.
  useEffect(() => {
    if (selectedId === null) return;
    let cancelled = false;

    setSummary("");
    setSavedAt(null);
    setModelAlias(null);
    setError(null);
    setProgress(null);

    getSummary(selectedId)
      .then((res) => {
        if (cancelled) return;
        setSummary(res.content);
        setSavedAt(res.updated_at);
        setModelAlias(res.model_alias);
      })
      .catch(() => {
        // 404 = henüz özet yok; bu bir hata değil, normal başlangıç durumu.
      });

    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  async function handleGenerate() {
    if (!user || selectedId === null) return;

    setGenerating(true);
    setError(null);
    setProgress(null);
    setSummary("");
    setSavedAt(null);

    let live = "";
    try {
      await createSummaryStream(
        { documentId: selectedId, userId: user.id },
        {
          onProgress: (detail) => setProgress(detail),
          onToken: (token) => {
            live += token;
            setSummary(live);
            // İlk token geldiğinde map adımı bitmiştir; ilerleme göstergesini
            // kaldırıp yerini canlı akan metne bırakıyoruz.
            setProgress(null);
          },
          onError: (detail) => setError(detail),
          onDone: () => {
            setSavedAt(new Date().toISOString());
            refreshStatus();
          },
        }
      );
    } catch (err) {
      setError(
        err instanceof Error && err.message
          ? err.message
          : "Özet üretilemedi. Foundry Local çalışıyor mu kontrol et."
      );
    } finally {
      setGenerating(false);
      setProgress(null);
    }
  }

  async function handleDelete() {
    if (selectedId === null) return;
    if (!confirm("Bu özet silinsin mi? Yeniden üretmen gerekir.")) return;
    await deleteSummary(selectedId);
    setSummary("");
    setSavedAt(null);
    setModelAlias(null);
    refreshStatus();
  }

  const selectedDoc = documents.find((d) => d.id === selectedId) ?? null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-white">Özet Çıkar</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Bir dosyanın tamamını özetle. Özet kaydedilir; quiz üretirken de bu özet
          kullanılır ve quiz belirgin biçimde daha hızlı üretilir.
        </p>
      </div>

      <DocumentList
        documents={documents}
        loading={documentsLoading}
        selectedId={selectedId}
        withSummary={withSummary}
        onSelect={setSelectedId}
      />

      {selectedDoc && (
        <Card className="p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-zinc-200">
                {selectedDoc.filename}
              </p>
              {savedAt && (
                <p className="mt-0.5 text-xs text-zinc-500">
                  Kayıtlı özet · {new Date(savedAt).toLocaleString("tr-TR")}
                  {modelAlias ? ` · ${modelAlias}` : ""}
                </p>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button onClick={handleGenerate} disabled={generating}>
                {generating ? (
                  <>
                    <Spinner /> Üretiliyor
                  </>
                ) : summary ? (
                  <>
                    <RefreshCw className="h-4 w-4" strokeWidth={1.75} /> Yeniden üret
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" strokeWidth={1.75} /> Özet çıkar
                  </>
                )}
              </Button>

              {summary && !generating && (
                <>
                  <Button
                    variant="ghost"
                    onClick={() => navigate("/quiz", { state: { documentId: selectedId } })}
                    title="Bu özetten quiz üret"
                  >
                    <Brain className="h-4 w-4" strokeWidth={1.75} /> Bu özetten quiz üret
                  </Button>
                  <Button variant="danger" onClick={handleDelete} title="Özeti sil">
                    <Trash2 className="h-4 w-4" strokeWidth={1.75} />
                  </Button>
                </>
              )}
            </div>
          </div>

          {progress && (
            <div className="mt-4 flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-950/50 px-3 py-2 text-xs text-zinc-400">
              <Spinner className="h-3.5 w-3.5" />
              {progress === "birleştiriliyor"
                ? "Bölüm özetleri birleştiriliyor..."
                : `Uzun doküman: bölüm ${progress} özetleniyor...`}
            </div>
          )}

          {error && (
            <div className="mt-4 rounded-lg border border-red-900/60 bg-red-950/30 px-3 py-2 text-xs text-red-300">
              {error}
            </div>
          )}

          {summary && (
            <div className="mt-4 border-t border-zinc-800 pt-4">
              <Markdown>{summary}</Markdown>
              {generating && (
                <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-indigo-400 align-middle" />
              )}
            </div>
          )}

          {!summary && !generating && !error && (
            <p className="mt-4 border-t border-zinc-800 pt-4 text-sm text-zinc-500">
              Bu dosya için henüz özet yok. "Özet çıkar" ile oluşturabilirsin.
            </p>
          )}
        </Card>
      )}
    </div>
  );
}

function DocumentList({
  documents,
  loading,
  selectedId,
  withSummary,
  onSelect,
}: {
  documents: DocumentOut[];
  loading: boolean;
  selectedId: number | null;
  withSummary: Set<number>;
  onSelect: (id: number) => void;
}) {
  if (loading) {
    return (
      <Card className="flex items-center gap-2 p-6 text-sm text-zinc-500">
        <Spinner /> Dosyalar yükleniyor...
      </Card>
    );
  }

  if (documents.length === 0) {
    return (
      <Card className="p-8 text-center text-sm text-zinc-500">
        Henüz yüklenmiş bir dosya yok — önce Dosya Yükle ekranından ekle.
      </Card>
    );
  }

  return (
    <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
      {documents.map((doc) => {
        const selected = doc.id === selectedId;
        return (
          <button
            key={doc.id}
            type="button"
            onClick={() => onSelect(doc.id)}
            className={`flex items-start gap-2.5 rounded-xl border px-3.5 py-3 text-left transition ${
              selected
                ? "border-indigo-500 bg-indigo-500/10"
                : "border-zinc-800 bg-zinc-900/40 hover:border-zinc-700"
            }`}
          >
            <FileText
              className={`mt-0.5 h-4 w-4 shrink-0 ${selected ? "text-indigo-400" : "text-zinc-600"}`}
              strokeWidth={1.75}
            />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm text-zinc-200">{doc.filename}</span>
              <span className="mt-0.5 block text-[11px] text-zinc-500">
                {withSummary.has(doc.id) ? "✓ özeti var" : "özet yok"}
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );
}
