import { useCallback, useEffect, useRef, useState } from "react";
import AskPage from "./AskPage";
import QuizPage from "./QuizPage";

const MIN_PCT = 25;
const MAX_PCT = 75;
const STORAGE_KEY = "ask-me:split-view-left-pct";

/**
 * Soru Sor ve Quiz'i yan yana, ortada sürüklenebilir bir ayırıcıyla gösterir
 * (kullanıcı isteği: "ekranı ikiye böl özelliğiyle soru sor ve quiz sayfası
 * yan yana da açılabilsin"). Her iki sayfa da normal /ask ve /quiz
 * rotalarında AYNEN kullanılan bileşenler — burada sadece iki tanesi aynı
 * anda, iki ayrı sütunda mount ediliyor. Tek incelik: her ikisinin de sağ
 * altta sabit bir "geçmiş" paneli var (bkz. HistoryPanel.tsx); aynı anda
 * göründüklerinde üst üste binmesinler diye birine "left" birine "right"
 * köşe veriyoruz (bkz. AskPage/QuizPage'in historyPanelSide prop'u).
 */
export default function SplitViewPage() {
  const [leftPct, setLeftPct] = useState<number>(() => {
    const saved = Number(localStorage.getItem(STORAGE_KEY));
    return saved >= MIN_PCT && saved <= MAX_PCT ? saved : 50;
  });
  const containerRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  const handleMove = useCallback((clientX: number) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setLeftPct(Math.min(MAX_PCT, Math.max(MIN_PCT, pct)));
  }, []);

  useEffect(() => {
    function onMouseMove(e: MouseEvent) {
      if (draggingRef.current) handleMove(e.clientX);
    }
    function onMouseUp() {
      if (draggingRef.current) {
        draggingRef.current = false;
        localStorage.setItem(STORAGE_KEY, String(leftPct));
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      }
    }
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
    // leftPct kasıtlı olarak dependency'de: onMouseUp anındaki en güncel
    // değeri localStorage'a yazabilmek için.
  }, [handleMove, leftPct]);

  function startDrag() {
    draggingRef.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }

  return (
    <div ref={containerRef} className="flex" style={{ minHeight: "calc(100vh - 8rem)" }}>
      <div style={{ width: `${leftPct}%` }} className="min-w-0 pr-3">
        <AskPage historyPanelSide="left" />
      </div>

      <div
        onMouseDown={startDrag}
        role="separator"
        aria-orientation="vertical"
        title="Sürükleyerek yeniden boyutlandır"
        className="group relative w-2 shrink-0 cursor-col-resize"
      >
        <div className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-zinc-800 transition group-hover:w-1 group-hover:bg-indigo-500/70" />
      </div>

      <div style={{ width: `${100 - leftPct}%` }} className="min-w-0 pl-3">
        <QuizPage historyPanelSide="right" />
      </div>
    </div>
  );
}
