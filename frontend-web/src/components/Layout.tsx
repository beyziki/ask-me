import { NavLink, Outlet, Navigate } from "react-router-dom";
import { FolderUp, MessageSquare, Brain, Code2, Columns2, GraduationCap, LogOut, ScrollText } from "lucide-react";
import { useAuth } from "../context/AuthContext";

const NAV_ITEMS = [
  { to: "/upload", label: "Dosya Yükle", Icon: FolderUp },
  { to: "/ask", label: "Soru Sor", Icon: MessageSquare },
  { to: "/summary", label: "Özet", Icon: ScrollText },
  { to: "/quiz", label: "Quiz", Icon: Brain },
  { to: "/split", label: "Yan Yana", Icon: Columns2 },
  { to: "/code", label: "Kod Analizi", Icon: Code2 },
];

export default function Layout() {
  const { user, logout } = useAuth();

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <header className="sticky top-0 z-10 w-full border-b border-zinc-800 bg-zinc-950/95 backdrop-blur">
        <div className="grid h-16 grid-cols-[auto_1fr_auto] items-center gap-6 px-6">
          <div className="flex items-center gap-2 justify-self-start">
            <GraduationCap className="h-5 w-5 text-indigo-400" strokeWidth={1.75} />
            <div className="leading-none">
              <p className="text-sm font-semibold text-white">Ask Me?</p>
              <p className="text-[11px] text-zinc-500">Study Assistant</p>
            </div>
          </div>

          <nav className="flex items-center gap-1 justify-self-center">
            {NAV_ITEMS.map(({ to, label, Icon }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition ${
                    isActive
                      ? "bg-indigo-600/15 text-indigo-300"
                      : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"
                  }`
                }
              >
                <Icon className="h-4 w-4" strokeWidth={1.75} />
                <span className="hidden sm:inline">{label}</span>
              </NavLink>
            ))}
          </nav>

          <div className="flex items-center gap-3 justify-self-end">
            <div className="hidden text-right sm:block">
              <p className="text-sm font-medium leading-none text-zinc-200">{user.username}</p>
              <p className="mt-0.5 text-[11px] text-zinc-500">
                {user.preferred_language === "tr" ? "Türkçe" : "English"}
              </p>
            </div>
            <button
              onClick={logout}
              title="Çıkış Yap"
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-zinc-800 text-zinc-400 transition hover:border-red-900/60 hover:bg-red-950/30 hover:text-red-300"
            >
              <LogOut className="h-4 w-4" strokeWidth={1.75} />
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 sm:py-10">
        <Outlet />
      </main>
    </div>
  );
}
