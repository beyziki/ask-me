import axios from "axios";

/**
 * FastAPI backend'ine giden tüm isteklerin ortak istemcisi.
 *
 * Backend'de gerçek bir oturum sistemi yok (bkz. backend/app/api/deps.py
 * içindeki TODO); kimlik doğrulama şu an basit bir "X-User-Id" header'ı ile
 * yapılıyor. Bu, proje kapsamında bilinçli olarak MVP seviyesinde bırakıldı
 * (Hafta 3'te JWT'ye geçilecek), bu yüzden burada da aynı mantığı koruyoruz.
 */
const baseURL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export const apiClient = axios.create({ baseURL });

/** Giriş yapmış kullanıcının localStorage anahtarı. AuthContext ile ORTAK
 * kullanılıyor (bkz. context/AuthContext.tsx) — iki yerde ayrı ayrı sabit
 * yazmamak için buradan export ediliyor. */
export const USER_STORAGE_KEY = "ask-me:user";

export function setUserId(userId: number | null) {
  if (userId === null) {
    delete apiClient.defaults.headers.common["X-User-Id"];
  } else {
    apiClient.defaults.headers.common["X-User-Id"] = String(userId);
  }
}

/**
 * Modül yüklenirken header'ı localStorage'daki kullanıcıdan kur.
 *
 * NEDEN: header'ı yalnızca AuthProvider'ın `useEffect`'i yazsaydı, sayfa
 * yenilendiğinde React ağacındaki ALT bileşenlerin effect'leri (ör.
 * DocumentsProvider'ın doküman listesini çekmesi) parent'ınkinden ÖNCE
 * çalıştığı için ilk istekler header'sız gidiyor ve backend 422
 * (eksik `X-User-Id`) dönüyordu. Modül seviyesinde bir kez kurmak bu yarışı
 * tamamen ortadan kaldırıyor: ilk render'dan bile önce hazır oluyor.
 */
function restoreUserIdFromStorage(): void {
  try {
    const raw = localStorage.getItem(USER_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw) as { id?: unknown };
    if (typeof parsed?.id === "number") setUserId(parsed.id);
  } catch {
    // Bozuk/eski bir kayıt varsa yok say; kullanıcı tekrar giriş yapar.
  }
}

restoreUserIdFromStorage();
