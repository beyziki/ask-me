import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { login, register } from "../api/endpoints";
import { useAuth } from "../context/AuthContext";
import { Button, Card, Input } from "../components/ui";

export default function LoginPage() {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [language, setLanguage] = useState<"tr" | "en">("tr");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const auth = useAuth();
  const navigate = useNavigate();

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const user =
        mode === "login"
          ? await login(username, password)
          : await register(username, password, language);
      auth.login(user);
      navigate("/", { replace: true });
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(detail ?? "Bir hata oluştu, tekrar dener misin?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-zinc-950 p-6">
      <Card className="w-full max-w-sm p-8 space-y-6 border-zinc-800">
        <div className="text-center space-y-1">
          <h1 className="text-2xl font-semibold text-white">Ask Me?</h1>
          <p className="text-sm text-zinc-500">Offline AI Study Assistant</p>
        </div>

        <div className="flex rounded-lg bg-zinc-950 border border-zinc-800 p-1 text-sm">
          <button
            type="button"
            onClick={() => setMode("login")}
            className={`flex-1 rounded-md py-2 transition ${
              mode === "login" ? "bg-indigo-600 text-white" : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            Giriş Yap
          </button>
          <button
            type="button"
            onClick={() => setMode("register")}
            className={`flex-1 rounded-md py-2 transition ${
              mode === "register" ? "bg-indigo-600 text-white" : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            Kayıt Ol
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-zinc-500">Kullanıcı adı</label>
            <Input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              autoComplete="username"
              placeholder="byzerdem"
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-zinc-500">Şifre</label>
            <Input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              placeholder="••••••••"
            />
          </div>

          {mode === "register" && (
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-zinc-500">Tercih edilen dil</label>
              <div className="flex gap-2">
                {(["tr", "en"] as const).map((lang) => (
                  <button
                    type="button"
                    key={lang}
                    onClick={() => setLanguage(lang)}
                    className={`flex-1 rounded-lg border py-2 text-sm transition ${
                      language === lang
                        ? "border-indigo-500 bg-indigo-500/10 text-indigo-300"
                        : "border-zinc-700 text-zinc-500 hover:border-zinc-600"
                    }`}
                  >
                    {lang === "tr" ? "Türkçe" : "English"}
                  </button>
                ))}
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-2.5 text-sm text-red-300">
              {error}
            </div>
          )}

          <Button type="submit" disabled={loading} className="w-full">
            {loading ? "Bekleniyor..." : mode === "login" ? "Giriş Yap" : "Hesap Oluştur"}
          </Button>
        </form>
      </Card>
    </div>
  );
}
