import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { listDocumentGroups, listDocuments } from "../api/endpoints";
import type { DocumentGroup, DocumentOut } from "../api/types";
import { useAuth } from "./AuthContext";

interface DocumentsContextValue {
  documents: DocumentOut[];
  groups: DocumentGroup[];
  loading: boolean;
  refresh: () => Promise<void>;
}

const DocumentsContext = createContext<DocumentsContextValue | null>(null);

/**
 * Doküman (ve doküman grubu) listesini tüm uygulama için TEK bir yerden,
 * önbellekli şekilde sağlar. Önceden her ekran (Dosya Yükle, Soru Sor,
 * Quiz, Kod Analizi) kendi başına `listDocuments()` çağırıyordu; bu da bir
 * ekrana her gidişte (ör. Quiz'den Soru Sor'a geçip geri dönünce) listenin
 * sıfırdan yeniden çekilmesine ve gereksiz bir "yükleniyor" beklemesine yol
 * açıyordu. Artık liste burada bir kez çekilip önbellekte tutuluyor;
 * ekranlar arası geçişte tekrar istek atılmıyor. Bir dosya/grup
 * yüklendiğinde/oluşturulduğunda/değiştirildiğinde (bkz. UploadPage)
 * `refresh()` elle çağrılarak önbellek güncelleniyor.
 */
export function DocumentsProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [groups, setGroups] = useState<DocumentGroup[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadedForUserId, setLoadedForUserId] = useState<number | null>(null);

  const refresh = useCallback(async () => {
    if (!user) return;
    setLoading(true);
    try {
      const [docs, grps] = await Promise.all([listDocuments(), listDocumentGroups()]);
      setDocuments(docs);
      setGroups(grps);
      setLoadedForUserId(user.id);
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    if (!user) {
      // Çıkış yapıldığında önbelleği temizle ki bir sonraki kullanıcı
      // öncekinin dosyalarını görmesin.
      setDocuments([]);
      setGroups([]);
      setLoadedForUserId(null);
      return;
    }
    if (loadedForUserId !== user.id) {
      refresh();
    }
    // loadedForUserId kasıtlı olarak dependency'de değil: refresh()
    // içeride onu güncelliyor, burada sadece "bu kullanıcı için daha önce
    // yüklendi mi" kontrolü yapıyoruz.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user, refresh]);

  return (
    <DocumentsContext.Provider value={{ documents, groups, loading, refresh }}>
      {children}
    </DocumentsContext.Provider>
  );
}

export function useDocuments(): DocumentsContextValue {
  const ctx = useContext(DocumentsContext);
  if (!ctx) throw new Error("useDocuments, DocumentsProvider içinde kullanılmalı");
  return ctx;
}
