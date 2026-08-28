import { useEffect, useMemo, useState } from "react";
import { FileCode2, Sparkles } from "lucide-react";
import { explainCode } from "../api/endpoints";
import { Button, Card, Spinner } from "../components/ui";
import Markdown from "../components/Markdown";
import { useDocuments } from "../context/DocumentsContext";

export default function CodePage() {
  const { documents: allDocuments } = useDocuments();
  const documents = useMemo(
    () => allDocuments.filter((d) => d.file_type === "code"),
    [allDocuments]
  );
  const [documentId, setDocumentId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<string | null>(null);

  // Doküman listesi artık DocumentsContext'ten (önbellekli) geliyor; burada
  // sadece henüz bir seçim yapılmamışsa ilk kod dosyasını varsayılan seçiyoruz.
  useEffect(() => {
    if (documentId === null && documents.length > 0) {
      setDocumentId(documents[0].id);
    }
  }, [documents, documentId]);

  async function handleExplain() {
    if (!documentId) return;
    setLoading(true);
    setError(null);
    setExplanation(null);
    try {
      const res = await explainCode(documentId);
      setExplanation(res.explanation);
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Kod analiz edilemedi.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-white">Kod Analizi</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Yüklediğin bir kod dosyasının ne yaptığını, ana fonksiyonlarını ve dikkat edilmesi
          gereken noktaları açıklat.
        </p>
      </div>

      <Card className="space-y-4 p-5">
        {documents.length === 0 ? (
          <p className="text-sm text-zinc-500">
            Henüz kod dosyası yüklenmedi. Dosya Yükle ekranından bir .py/.js/.ts/.java vb. dosya
            ekle.
          </p>
        ) : (
          <>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-zinc-500">Kod dosyası</label>
              <select
                value={documentId ?? ""}
                onChange={(e) => setDocumentId(Number(e.target.value))}
                className="w-full rounded-lg border border-zinc-700 bg-zinc-950/60 px-3 py-2.5 text-sm text-zinc-100 outline-none focus:border-indigo-500"
              >
                {documents.map((doc) => (
                  <option key={doc.id} value={doc.id}>
                    {doc.filename}
                  </option>
                ))}
              </select>
            </div>

            <Button onClick={handleExplain} disabled={loading} className="w-full sm:w-auto">
              {loading ? (
                <>
                  <Spinner className="h-4 w-4" /> Analiz ediliyor...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" strokeWidth={1.75} /> Kodu Açıkla
                </>
              )}
            </Button>
          </>
        )}

        {error && (
          <div className="rounded-lg border border-red-900/60 bg-red-950/30 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}
      </Card>

      {explanation && (
        <Card className="space-y-3 p-6">
          <div className="flex items-center gap-2 text-xs font-medium text-zinc-500">
            <FileCode2 className="h-3.5 w-3.5" strokeWidth={1.75} />
            Açıklama
          </div>
          <Markdown>{explanation}</Markdown>
        </Card>
      )}
    </div>
  );
}
