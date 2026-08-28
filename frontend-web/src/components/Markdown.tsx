import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Model cevaplarını markdown olarak render eder.
 *
 * NEDEN: cevaplar eskiden düz metin olarak (`whitespace-pre-wrap`)
 * gösteriliyordu — model başlık/liste/kalın üretse bile ekranda ham
 * `## Başlık` ve `**kalın**` olarak görünüyor, uzun cevaplar da tek bir
 * duvar hâlinde okunması zor kalıyordu.
 *
 * Stiller `components` ile ELEMENT ELEMENT veriliyor; `@tailwindcss/typography`
 * (prose) eklentisine bağımlılık eklemeden koyu temayla uyumlu, sohbet
 * balonu içinde dengeli boşluklu bir görünüm elde ediyoruz.
 *
 * `remark-gfm` tablo, görev listesi ve şerit (~~üstü çizili~~) desteği ekliyor;
 * ders notlarından üretilen cevaplarda tablolar sık çıkıyor.
 *
 * GÜVENLİK: react-markdown varsayılan olarak ham HTML'i render ETMEZ
 * (`rehype-raw` eklenmedi), yani doküman içeriğinden gelen HTML enjekte
 * edilemez.
 *
 * AKIŞ (streaming) NOTU: cevap token token geldiği için ara karelerde
 * yarım markdown (ör. henüz kapanmamış `**`) oluşabilir; react-markdown bunu
 * düz metin olarak gösterip bir sonraki token'da kendiliğinden düzeltiyor,
 * ekstra bir önleme gerek yok.
 */
export default function Markdown({ children }: { children: string }) {
  return (
    <div className="space-y-3 text-sm leading-relaxed text-zinc-200">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="mt-1 text-base font-semibold text-white">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="mt-1 text-[15px] font-semibold text-white">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="mt-1 text-sm font-semibold text-zinc-100">{children}</h3>
          ),
          p: ({ children }) => <p>{children}</p>,
          ul: ({ children }) => (
            <ul className="list-disc space-y-1 pl-5 marker:text-zinc-600">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal space-y-1 pl-5 marker:text-zinc-600">{children}</ol>
          ),
          li: ({ children }) => <li className="pl-0.5">{children}</li>,
          strong: ({ children }) => (
            <strong className="font-semibold text-white">{children}</strong>
          ),
          em: ({ children }) => <em className="italic text-zinc-300">{children}</em>,
          a: ({ children, href }) => (
            <a
              href={href}
              target="_blank"
              rel="noreferrer"
              className="text-indigo-400 underline underline-offset-2 hover:text-indigo-300"
            >
              {children}
            </a>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-zinc-700 pl-3 text-zinc-400">
              {children}
            </blockquote>
          ),
          hr: () => <hr className="border-zinc-800" />,
          code: ({ children, className }) => {
            // react-markdown, blok kodda dil sınıfı (`language-*`) veriyor;
            // satır içi kodda vermiyor. İkisini bu şekilde ayırıyoruz.
            const isBlock = Boolean(className);
            if (isBlock) {
              return (
                <code className="block overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950/70 p-3 font-mono text-xs text-zinc-300">
                  {children}
                </code>
              );
            }
            return (
              <code className="rounded bg-zinc-800/80 px-1 py-0.5 font-mono text-[0.85em] text-indigo-200">
                {children}
              </code>
            );
          },
          // `pre` kendi kutusunu çizmiyor; kutu yukarıdaki blok `code`'da.
          pre: ({ children }) => <pre className="m-0">{children}</pre>,
          table: ({ children }) => (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-xs">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="border border-zinc-800 bg-zinc-900/60 px-2 py-1 text-left font-semibold text-zinc-200">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border border-zinc-800 px-2 py-1 align-top">{children}</td>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
