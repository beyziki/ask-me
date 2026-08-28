import { useMemo, useRef, useState } from "react";
import {
  FileText,
  FileCode2,
  FileType,
  File as FileIcon,
  UploadCloud,
  FolderPlus,
  Trash2,
  X,
} from "lucide-react";
import {
  assignDocumentGroup,
  createDocumentGroup,
  deleteDocument,
  deleteDocumentGroup,
  uploadDocument,
} from "../api/endpoints";
import type { DocumentOut } from "../api/types";
import { Card, Spinner } from "../components/ui";
import { useDocuments } from "../context/DocumentsContext";

const FILE_TYPE_LABEL: Record<string, string> = {
  pdf: "PDF",
  markdown: "Markdown",
  code: "Kod",
};

const FILE_TYPE_ICON: Record<string, typeof FileIcon> = {
  pdf: FileType,
  markdown: FileText,
  code: FileCode2,
};

// Gruplanmamış dosyalar için sahte bir grup id'si; gerçek grup id'leri
// backend'de her zaman pozitif tam sayı olduğu için 0 ile çakışmaz.
const UNGROUPED_KEY = "ungrouped";

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString("tr-TR", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export default function UploadPage() {
  const {
    documents,
    groups,
    loading: loadingList,
    refresh: refreshDocuments,
  } = useDocuments();
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Yeni yüklenen dosyaların hangi gruba gideceği; null = Grupsuz.
  const [uploadGroupId, setUploadGroupId] = useState<number | null>(null);
  const [newGroupName, setNewGroupName] = useState("");
  const [creatingGroup, setCreatingGroup] = useState(false);
  // Bir dokümanın grup dropdown'ı değiştirilirken (PATCH isteği sürerken)
  // o dokümanın id'sini tutuyoruz; yalnızca o kart "güncelleniyor" görünsün diye.
  const [reassigningId, setReassigningId] = useState<number | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    setError(null);
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        await uploadDocument(file, uploadGroupId);
      }
      await refreshDocuments();
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Dosya yüklenirken bir hata oluştu.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleCreateGroup() {
    const name = newGroupName.trim();
    if (!name) return;
    setCreatingGroup(true);
    setError(null);
    try {
      const group = await createDocumentGroup(name);
      setNewGroupName("");
      await refreshDocuments();
      // Az önce oluşturulan grubu yükleme hedefi olarak da otomatik seçelim;
      // kullanıcı genelde grubu tam da içine dosya koymak için oluşturur.
      setUploadGroupId(group.id);
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Grup oluşturulurken bir hata oluştu.");
    } finally {
      setCreatingGroup(false);
    }
  }

  async function handleDeleteGroup(groupId: number) {
    setError(null);
    try {
      await deleteDocumentGroup(groupId);
      if (uploadGroupId === groupId) setUploadGroupId(null);
      await refreshDocuments();
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Grup silinirken bir hata oluştu.");
    }
  }

  async function handleReassign(doc: DocumentOut, groupId: number | null) {
    setReassigningId(doc.id);
    setError(null);
    try {
      await assignDocumentGroup(doc.id, groupId);
      await refreshDocuments();
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Doküman grubu güncellenirken bir hata oluştu.");
    } finally {
      setReassigningId(null);
    }
  }

  async function handleDelete(doc: DocumentOut) {
    // Yanlışlıkla eklenen bir dosyayı silmek geri alınamaz (chunk'lar ve
    // FAISS vektörleri de gidiyor), bu yüzden tek tıkla silmek yerine
    // basit bir onay istiyoruz.
    if (!window.confirm(`"${doc.filename}" silinsin mi? Bu işlem geri alınamaz.`)) return;
    setDeletingId(doc.id);
    setError(null);
    try {
      await deleteDocument(doc.id);
      await refreshDocuments();
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Dosya silinirken bir hata oluştu.");
    } finally {
      setDeletingId(null);
    }
  }

  // Dosyaları gruplarına göre kova (bucket) haline getiriyoruz; gruplar
  // `groups`'taki sırayla, gruplanmamış dosyalar en sonda gösteriliyor.
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

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-white">Dosya Yükle</h1>
        <p className="mt-1 text-sm text-zinc-500">
          PDF, Markdown veya kod dosyalarını yükle; Soru Sor ve Quiz ekranları bu kaynakları
          kullanacak. İstersen dosyaları gruplayarak (ör. ders/konu bazlı) daha sonra Soru
          Sor'da yalnızca ilgili grubu seçebilirsin.
        </p>
      </div>

      <Card className="space-y-3 p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-medium text-zinc-300">Gruplar</h2>
        </div>

        {groups.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {groups.map((g) => (
              <span
                key={g.id}
                className="flex items-center gap-1.5 rounded-full border border-zinc-800 bg-zinc-900 py-1 pl-3 pr-1.5 text-xs text-zinc-300"
              >
                {g.name}
                <button
                  type="button"
                  onClick={() => handleDeleteGroup(g.id)}
                  title="Grubu sil (dosyalar silinmez, grupsuz kalır)"
                  className="rounded-full p-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-red-400"
                >
                  <X className="h-3 w-3" strokeWidth={2} />
                </button>
              </span>
            ))}
          </div>
        )}

        <div className="flex gap-2">
          <input
            value={newGroupName}
            onChange={(e) => setNewGroupName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleCreateGroup();
              }
            }}
            placeholder="Yeni grup adı (ör. Bilgisayar Ağları)"
            className="w-full rounded-lg border border-zinc-700 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 outline-none focus:border-indigo-500"
          />
          <button
            type="button"
            onClick={handleCreateGroup}
            disabled={creatingGroup || !newGroupName.trim()}
            className="flex shrink-0 items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-indigo-500 disabled:opacity-50"
          >
            {creatingGroup ? <Spinner className="h-4 w-4" /> : <FolderPlus className="h-4 w-4" strokeWidth={1.75} />}
            Oluştur
          </button>
        </div>
      </Card>

      <Card
        className={`border-2 border-dashed p-10 text-center transition ${
          isDragging ? "border-indigo-500 bg-indigo-500/5" : "border-zinc-800"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
      >
        {uploading ? (
          <div className="flex flex-col items-center gap-3 text-zinc-300">
            <Spinner className="h-6 w-6" />
            <p className="text-sm">Yükleniyor ve işleniyor (chunking + embedding)...</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <UploadCloud className="h-8 w-8 text-zinc-500" strokeWidth={1.5} />
            <p className="text-sm text-zinc-300">
              Dosyayı buraya sürükle bırak veya
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="ml-1 text-indigo-400 hover:text-indigo-300 underline underline-offset-2"
              >
                bilgisayardan seç
              </button>
            </p>
            <p className="text-xs text-zinc-600">PDF, .md, .py, .js, .ts, .java, .cpp vb.</p>

            {groups.length > 0 && (
              <div className="mt-1 flex items-center gap-2 text-xs text-zinc-500">
                <span>Hedef grup:</span>
                <select
                  value={uploadGroupId ?? ""}
                  onChange={(e) => setUploadGroupId(e.target.value ? Number(e.target.value) : null)}
                  onClick={(e) => e.stopPropagation()}
                  className="rounded-lg border border-zinc-700 bg-zinc-950/60 px-2 py-1 text-xs text-zinc-200 outline-none focus:border-indigo-500"
                >
                  <option value="">Grupsuz</option>
                  {groups.map((g) => (
                    <option key={g.id} value={g.id}>
                      {g.name}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </Card>

      {error && (
        <div className="rounded-lg border border-red-900/60 bg-red-950/30 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      <div>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-medium text-zinc-300">Yüklenen Dosyalar</h2>
          {!loadingList && (
            <span className="text-xs text-zinc-600">{documents.length} dosya</span>
          )}
        </div>

        {loadingList ? (
          <div className="flex items-center gap-2 text-sm text-zinc-500">
            <Spinner className="h-4 w-4" /> Yükleniyor...
          </div>
        ) : documents.length === 0 ? (
          <Card className="p-8 text-center text-sm text-zinc-500">
            Henüz dosya yüklenmedi. Başlamak için yukarıdan bir dosya seç.
          </Card>
        ) : (
          <div className="space-y-6">
            {buckets.map((bucket) => (
              <div key={bucket.key} className="space-y-2">
                <p className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                  {bucket.label} <span className="text-zinc-700">· {bucket.docs.length}</span>
                </p>
                {bucket.docs.map((doc) => {
                  const Icon = FILE_TYPE_ICON[doc.file_type] ?? FileIcon;
                  return (
                    <Card
                      key={doc.id}
                      className="flex items-center gap-3 px-4 py-3 hover:border-zinc-700 transition"
                    >
                      <Icon className="h-5 w-5 shrink-0 text-zinc-500" strokeWidth={1.75} />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium text-zinc-200">{doc.filename}</p>
                        <p className="text-xs text-zinc-600">{formatDate(doc.uploaded_at)}</p>
                      </div>
                      <span className="shrink-0 rounded-full border border-zinc-800 bg-zinc-900 px-2.5 py-1 text-xs text-zinc-400">
                        {FILE_TYPE_LABEL[doc.file_type] ?? doc.file_type}
                      </span>
                      {groups.length > 0 && (
                        <select
                          value={doc.group_id ?? ""}
                          disabled={reassigningId === doc.id}
                          onChange={(e) =>
                            handleReassign(doc, e.target.value ? Number(e.target.value) : null)
                          }
                          title="Grubu değiştir"
                          className="shrink-0 rounded-lg border border-zinc-800 bg-zinc-950/60 px-2 py-1.5 text-xs text-zinc-300 outline-none focus:border-indigo-500 disabled:opacity-50"
                        >
                          <option value="">Grupsuz</option>
                          {groups.map((g) => (
                            <option key={g.id} value={g.id}>
                              {g.name}
                            </option>
                          ))}
                        </select>
                      )}
                      <button
                        type="button"
                        onClick={() => handleDelete(doc)}
                        disabled={deletingId === doc.id}
                        title="Dosyayı sil"
                        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-zinc-600 transition hover:bg-red-950/40 hover:text-red-400 disabled:opacity-50"
                      >
                        {deletingId === doc.id ? (
                          <Spinner className="h-3.5 w-3.5" />
                        ) : (
                          <Trash2 className="h-3.5 w-3.5" strokeWidth={1.75} />
                        )}
                      </button>
                    </Card>
                  );
                })}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
