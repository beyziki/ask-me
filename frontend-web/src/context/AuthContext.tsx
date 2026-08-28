import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { USER_STORAGE_KEY, setUserId } from "../api/client";
import type { User } from "../api/types";

interface AuthContextValue {
  user: User | null;
  login: (user: User) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    try {
      const raw = localStorage.getItem(USER_STORAGE_KEY);
      return raw ? (JSON.parse(raw) as User) : null;
    } catch {
      return null;
    }
  });

  // ÖNEMLİ: header ve localStorage yazımı `login`/`logout` içinde SENKRON
  // yapılıyor, bir `useEffect` içinde DEĞİL.
  //
  // NEDEN: React, effect'leri çocuktan ebeveyne doğru çalıştırır. Header'ı
  // yalnızca buradaki effect yazsaydı, giriş yapıldığı anda ALT bileşen olan
  // DocumentsProvider'ın effect'i (doküman listesini çeken) ÖNCE çalışır ve
  // istek `X-User-Id` header'ı olmadan gider — backend de 422 (eksik zorunlu
  // header) döner. Gerçekte gözlemlenen davranış tam olarak buydu: giriş 200
  // OK dönüyor, hemen ardından `GET /documents` ve `GET /documents/groups`
  // 422 alıyordu. Senkron yazım bu yarışı ortadan kaldırıyor.
  const login = (u: User) => {
    localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(u));
    setUserId(u.id);
    setUser(u);
  };

  const logout = () => {
    localStorage.removeItem(USER_STORAGE_KEY);
    setUserId(null);
    setUser(null);
  };

  // Güvenlik ağı: kullanıcı state'i başka bir yoldan değişirse (ör. ileride
  // eklenecek bir "oturum yenileme") header ile localStorage yine senkron
  // kalsın. Yukarıdaki senkron yazımla birlikte idempotent çalışıyor.
  useEffect(() => {
    setUserId(user?.id ?? null);
    if (user) {
      localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(user));
    } else {
      localStorage.removeItem(USER_STORAGE_KEY);
    }
  }, [user]);

  return (
    <AuthContext.Provider value={{ user, login, logout }}>{children}</AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth, AuthProvider içinde kullanılmalı");
  return ctx;
}
