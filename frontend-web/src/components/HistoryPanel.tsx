import { useState } from "react";
import { History, Trash2, Plus, X } from "lucide-react";
import { Card } from "./ui";

interface HistoryItem {
  id: string;
  label: string;
  meta: string;
}

interface HistoryPanelProps {
  title: string;
  items: HistoryItem[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onRemove: (id: string) => void;
  onNew?: () => void;
  emptyText: string;
  /** Panelin ekranın hangi alt köşesinde duracağı. Varsayılan "right" — tek
   * başına (Soru Sor ya da Quiz) tam sayfa açıkken doğru köşe. Split View'da
   * (bkz. SplitViewPage.tsx) iki panel aynı anda göründüğü için biri "left"
   * biri "right" veriliyor, aksi halde ikisi de aynı köşede üst üste biner. */
  side?: "left" | "right";
}

/** Alt köşede sabit duran, açılıp kapanan geçmiş paneli.
 * Soru Sor (sohbet geçmişi) ve Quiz (quiz geçmişi) ekranlarında ortak
 * kullanılıyor; içerik alanının yanına sütun olarak eklemek yerine ekranın
 * köşesinde kalıyor, böylece ana içerik tüm genişliği kullanabiliyor. */
export default function HistoryPanel({
  title,
  items,
  activeId,
  onSelect,
  onRemove,
  onNew,
  emptyText,
  side = "right",
}: HistoryPanelProps) {
  const [open, setOpen] = useState(false);
  const sideClasses = side === "left" ? "left-6 items-start" : "right-6 items-end";

  return (
    <div className={`fixed bottom-6 z-30 flex flex-col gap-3 ${sideClasses}`}>
      {open && (
        <Card className="max-h-[60vh] w-80 overflow-y-auto p-3">
          <div className="mb-2 flex items-center justify-between px-1">
            <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-400">
              <History className="h-3.5 w-3.5" strokeWidth={1.75} />
              {title}
            </div>
            <div className="flex items-center gap-1">
              {onNew && (
                <button
                  type="button"
                  onClick={onNew}
                  title="Yeni"
                  className="rounded-md p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
                >
                  <Plus className="h-3.5 w-3.5" strokeWidth={1.75} />
                </button>
              )}
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-md p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
              >
                <X className="h-3.5 w-3.5" strokeWidth={1.75} />
              </button>
            </div>
          </div>

          {items.length === 0 ? (
            <p className="px-1 py-3 text-xs text-zinc-600">{emptyText}</p>
          ) : (
            <div className="space-y-1.5">
              {items.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onSelect(item.id)}
                  className={`group flex w-full items-center justify-between gap-2 rounded-lg border px-3 py-2.5 text-left text-xs transition ${
                    item.id === activeId
                      ? "border-indigo-500/50 bg-indigo-500/10 text-indigo-200"
                      : "border-zinc-800 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200"
                  }`}
                >
                  <span className="min-w-0">
                    <span className="block truncate font-medium">{item.label}</span>
                    <span className="text-[11px] text-zinc-600">{item.meta}</span>
                  </span>
                  <Trash2
                    className="h-3.5 w-3.5 shrink-0 text-zinc-600 opacity-0 transition hover:text-red-400 group-hover:opacity-100"
                    strokeWidth={1.75}
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemove(item.id);
                    }}
                  />
                </button>
              ))}
            </div>
          )}
        </Card>
      )}

      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="relative flex h-12 w-12 items-center justify-center rounded-full bg-indigo-600 text-white shadow-lg shadow-black/40 transition hover:bg-indigo-500"
      >
        <History className="h-5 w-5" strokeWidth={1.75} />
        {items.length > 0 && (
          <span className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full bg-zinc-950 text-[10px] font-medium text-zinc-200 ring-2 ring-zinc-950">
            {items.length}
          </span>
        )}
      </button>
    </div>
  );
}
