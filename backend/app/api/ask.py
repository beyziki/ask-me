"""Hybrid RAG ile soru-cevap uç noktası."""
import json
import logging

from langdetect import detect, LangDetectException
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db, get_current_user_id
from backend.app.core.config import settings
from backend.app.db.base import SessionLocal
from backend.app.db.models import Chunk, Document
from backend.app.models.schemas import AskRequest, AskResponse, SourceRef
from backend.app.services.llm import (
    DegenerateOutput,
    EmptyAnswer,
    FoundryNotAvailable,
    GpuContextLost,
    generate_answer,
    generate_answer_stream,
)
from backend.app.services.rag import bm25_search, rrf_merge, semantic_search

router = APIRouter(prefix="/ask", tags=["ask"])
logger = logging.getLogger(__name__)


# Türkçe'ye özgü harfler. `langdetect` istatistiksel bir model olduğu için
# kısa metinlerde (ve sorular genelde kısadır) tutarsız: "Turing makinesi
# nedir?" gibi bir soru rahatlıkla "en" olarak etiketlenebiliyordu -- bu da
# İngilizce sistem prompt'unun seçilmesine ve modelin Türkçe soruya
# İngilizce cevap vermesine yol açıyordu. Bu harflerden biri geçiyorsa metin
# kesinlikle Türkçedir; bu ucuz kontrol istatistiksel tahminden önce çalışır.
_TURKISH_CHARS = set("çğıİöşüÇĞÖŞÜ")

# Bu uzunluğun altındaki metinlerde langdetect'e hiç güvenmiyoruz;
# uygulamanın birincil dili olan Türkçeye düşüyoruz.
_MIN_CHARS_FOR_LANGDETECT = 25


def _detect_language(text: str) -> str:
    """Sorunun dilini belirler (yalnızca kullanıcı açıkça bir dil
    seçmediğinde çağrılır; bkz. `AskRequest.language`)."""
    if any(ch in _TURKISH_CHARS for ch in text):
        return "tr"
    if len(text.strip()) < _MIN_CHARS_FOR_LANGDETECT:
        return "tr"
    try:
        lang = detect(text)
    except LangDetectException:
        return "tr"
    return "en" if lang == "en" else "tr"


def _dedupe_by_content(hits: list, chunk_id_to_text: dict[int, str]) -> list:
    """Aynı fiziksel dosyanın birden fazla kez yüklenmesi (ör. kullanıcı
    yanlışlıkla aynı PDF'i tekrar tekrar yükleyince, bkz. Dosya Yükle
    ekranındaki yeni silme özelliği) chunk tablosunda İÇERİĞİ birebir aynı
    ama `chunk_id`'si (ve muhtemelen `document_id`'si) farklı satırlara yol
    açabiliyor. `hybrid_merge` zaten `chunk_id` bazında dedupe ediyor, ama
    bu tür kopyalar farklı `chunk_id`'lere sahip olduğu için orada
    yakalanmıyor — hem LLM'e giden bağlamda gereksiz yer kaplıyor hem de
    kullanıcıya "Kaynaklar" listesinde birebir aynı metnin art arda
    tekrarlandığı, kafa karıştırıcı bir görünüm olarak çıkıyor (gözlemlenen
    bir kullanıcı ekran görüntüsü).

    Burada, sırayı (zaten skora göre azalan) koruyarak İÇERİĞİ daha önce
    görülmüş olan hit'leri eliyoruz — hem context'e giden metni tekrarsız
    hem de kaynak listesini temiz tutuyor.
    """
    seen: set[str] = set()
    deduped = []
    for hit in hits:
        text = chunk_id_to_text.get(hit.chunk_id)
        if text is None or text in seen:
            continue
        seen.add(text)
        deduped.append(hit)
    return deduped


def _cap_context(chunks: list[str], max_chars: int) -> list[str]:
    """Context'i toplam karakter üst sınırına göre kırpar.

    Prefill (context'i okuma) süresi context uzunluğuyla doğru orantılı;
    ölçüsüz büyüyen bir context, cevabın ilk token'ının ekrana gelmesini
    gözle görülür biçimde geciktiriyor. Sıra zaten alakaya göre azalan
    olduğu için baştan alıp sınıra gelince kesiyoruz (yarım parça
    göndermemek için parçayı ya tamamen alıyoruz ya hiç).
    """
    kept: list[str] = []
    total = 0
    for chunk in chunks:
        if total + len(chunk) > max_chars and kept:
            break
        kept.append(chunk)
        total += len(chunk)
    return kept


def _retrieve(
    payload: AskRequest, user_id: int, db: Session
) -> tuple[list[str], list[SourceRef], str, bool]:
    """Hybrid RAG ile ilgili parçaları bulur; hem `/ask` hem `/ask/stream`
    tarafından kullanılan ortak alma (retrieval) adımı.

    Dönüş: (context_chunks, sources, language, has_context). Bu adım LLM'e
    hiç gitmeden tamamlanır, bu yüzden hem normal hem streaming uç noktasında
    aynı şekilde (LLM çağrısından ÖNCE, session açıkken) çalıştırılabilir.

    `has_context=False`, aramada yeterince ilgili hiçbir parça bulunamadığı
    anlamına gelir (bkz. `settings.min_relevance_score`); bu durumda
    kullanıcıya alakasız kaynaklardan zorlama bir cevap üretmek yerine
    modelden genel bilgiyle, bunu açıkça belirterek cevap vermesi isteniyor
    (bkz. services/llm.py:build_prompt).
    """
    query = db.query(Chunk).filter(Chunk.owner_id == user_id)
    if payload.document_ids:
        query = query.filter(Chunk.document_id.in_(payload.document_ids))
    chunks = query.all()

    row_to_chunk_id = {c.vector_row: c.id for c in chunks if c.vector_row is not None}
    chunk_id_to_text = {c.id: c.content for c in chunks}
    chunk_id_to_obj = {c.id: c for c in chunks}

    # İhtiyacımız olan tüm veriyi yukarıda çektik; aşağıdaki LLM çağrısı
    # (Foundry Local) soğuk başlangıçta veya üretim sırasında onlarca
    # saniye sürebiliyor. Bu süre boyunca havuzdan bir bağlantıyı işgal
    # etmemek için session'ı burada kapatıyoruz (aksi halde kullanıcı
    # yanıt gecikince tekrar soru gönderdiğinde QueuePool hızla tükeniyordu).
    db.close()

    language = payload.language or _detect_language(payload.question)

    sem_hits = semantic_search(user_id, payload.question, settings.top_k_semantic, row_to_chunk_id)
    bm25_hits = bm25_search(chunk_id_to_text, payload.question, settings.top_k_bm25)

    # En yüksek semantic benzerlik, "bu soru gerçekten dosyalarla ilgili mi?"
    # sorusunun tek anlamlı göstergesi (BM25 skorları normalize edildiği için
    # bu amaçla kullanılamaz). Eşiğin altındaysa RAG'ın bulduğu parçalar
    # yalnızca "en az alakasız" olanlardır -- bunları context diye modele
    # vermek, kullanıcının şikâyet ettiği "konuyla ilgisi olmayan cevap"
    # davranışının ana kaynağıydı.
    best_semantic = max((h.score for h in sem_hits), default=0.0)
    has_context = best_semantic >= settings.min_relevance_score

    # Eşiğin altında kalmak KULLANICIYA "dosyalarda ilgili bilgi yok" olarak
    # görünüyor, ama neden kaldığı (gerçekten alakasız bir soru mu, yoksa
    # eşik bu korpus/model için mi yüksek) dışarıdan anlaşılmıyordu.
    # Gerçek skoru loga yazıyoruz: `.env`'deki MIN_RELEVANCE_SCORE'u
    # ayarlamak isteyen biri artık bir sayıya bakarak karar verebiliyor.
    logger.info(
        "retrieval: kapsam=%d chunk, semantic aday=%d, en iyi skor=%.3f, "
        "eşik=%.2f -> has_context=%s",
        len(chunk_id_to_text),
        len(sem_hits),
        best_semantic,
        settings.min_relevance_score,
        has_context,
    )

    # Önce TÜM sıralanmış adaylar üzerinde içerik bazlı dedupe yapıp SONRA
    # kırpıyoruz (bkz. _dedupe_by_content) — böylece bir kopya yüzünden bir
    # yer boşa gitmiyor, onun yerine sıradaki farklı içerikli aday o yeri
    # dolduruyor.
    merged = _dedupe_by_content(rrf_merge(sem_hits, bm25_hits), chunk_id_to_text)[
        : settings.max_context_chunks
    ]

    context_chunks = [chunk_id_to_text[h.chunk_id] for h in merged if h.chunk_id in chunk_id_to_text]
    context_chunks = _cap_context(context_chunks, settings.max_context_chars)
    if not context_chunks:
        has_context = False

    # Kaynak (source) bilgisi için kısa ömürlü, ayrı bir session açıyoruz.
    # Kaynaklar YALNIZCA gerçekten context olarak kullanılan parçalar için
    # gösteriliyor: eskiden LLM'e hiç gitmemiş parçalar da "Kaynaklar"
    # listesinde görünüyordu, bu da cevapla kaynak listesinin uyuşmamasına
    # yol açıyordu. İlgili parça yoksa (has_context=False) hiç kaynak
    # göstermiyoruz -- cevap zaten dosyalardan gelmiyor.
    sources: list[SourceRef] = []
    if has_context:
        used_chunk_ids = [
            h.chunk_id
            for h in merged
            if h.chunk_id in chunk_id_to_text
            and chunk_id_to_text[h.chunk_id] in set(context_chunks)
        ]
        fresh_db = SessionLocal()
        try:
            document_ids = {
                chunk_id_to_obj[cid].document_id
                for cid in used_chunk_ids
                if cid in chunk_id_to_obj
            }
            # Eskiden her hit için ayrı bir SELECT atılıyordu (N+1). Tek
            # sorguda toplayıp sözlükten okuyoruz.
            docs = (
                fresh_db.query(Document).filter(Document.id.in_(document_ids)).all()
                if document_ids
                else []
            )
            filenames = {doc.id: doc.filename for doc in docs}
        finally:
            fresh_db.close()

        for chunk_id in used_chunk_ids:
            chunk = chunk_id_to_obj.get(chunk_id)
            if not chunk:
                continue
            sources.append(
                SourceRef(
                    document_id=chunk.document_id,
                    filename=filenames.get(chunk.document_id, ""),
                    chunk_index=chunk.chunk_index,
                    snippet=chunk.content[:200],
                )
            )

    return context_chunks, sources, language, has_context


@router.post("", response_model=AskResponse)
def ask(
    payload: AskRequest,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Tek seferde (streaming olmayan) cevap. Programatik kullanım ve
    testler için korunuyor; asıl arayüz artık `/ask/stream`'i kullanıyor
    (bkz. aşağıdaki `ask_stream`)."""
    context_chunks, sources, language, has_context = _retrieve(payload, user_id, db)
    try:
        answer = generate_answer(payload.question, context_chunks, language, has_context)
    except (FoundryNotAvailable, DegenerateOutput, EmptyAnswer, GpuContextLost) as exc:
        # 503: istemcinin isteği değil, yerel model/servis kaynaklı geçici
        # bir sorun (bkz. llm.py:_RepetitionGuard, _is_blank/EmptyAnswer ve
        # FoundryNotAvailable).
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AskResponse(answer=answer, sources=sources, has_context=has_context)


def _sse(event: dict) -> str:
    """Tek bir Server-Sent Events (SSE) olayını `data: <json>\\n\\n` formatına
    çevirir. Basit tutmak için özel bir `event:` alanı kullanmıyoruz; olay
    tipini JSON gövdesindeki `type` alanıyla ayırt ediyoruz (bkz.
    frontend-web/src/api/endpoints.ts:askQuestionStream)."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/stream")
def ask_stream(
    payload: AskRequest,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Cevabı token token (SSE) üreten uç nokta.

    Retrieval (hybrid RAG) adımı LLM üretimi başlamadan ÖNCE tamamlanıp
    kaynaklar hemen ilk olayla gönderiliyor — kaynaklar zaten LLM'in
    ürettiği metne değil, RAG aramasının sonucuna bağlı, bu yüzden frontend
    "şu kaynaklardan yararlanıyorum" bilgisini cevap yazılmaya başlamadan
    önce gösterebilir. Olay sırası: `sources` -> (birden çok) `token` ->
    `done` (veya bir hata olursa `error`).
    """
    context_chunks, sources, language, has_context = _retrieve(payload, user_id, db)

    def event_stream():
        # `has_context`: hybrid RAG yeterince ilgili bir parça bulabildi mi?
        # False ise cevap kullanıcının dosyalarından değil, modelin genel
        # bilgisinden geliyor -- arayüz bunu kullanıcıya açıkça gösteriyor
        # (bkz. frontend-web/src/pages/AskPage.tsx).
        yield _sse(
            {
                "type": "sources",
                "sources": [s.model_dump() for s in sources],
                "has_context": has_context,
            }
        )
        try:
            for piece in generate_answer_stream(
                payload.question, context_chunks, language, has_context
            ):
                yield _sse({"type": "token", "content": piece})
        except (FoundryNotAvailable, GpuContextLost) as exc:
            yield _sse({"type": "error", "detail": str(exc)})
            return
        except DegenerateOutput as exc:
            # Beklenmedik bir bug değil, yerel modelin bilinen bir arıza
            # modu (bkz. llm.py:_RepetitionGuard); bu yüzden diğer
            # `except Exception` dalının aksine `logger.exception` ile
            # gürültü yapmıyoruz — sadece bilgi amaçlı `warning`.
            logger.warning("Degenerate model output: %s", exc)
            yield _sse({"type": "error", "detail": str(exc)})
            return
        except EmptyAnswer as exc:
            # Yine bilinen bir arıza modu (bkz. llm.py:EmptyAnswer): model
            # görünür hiçbir metin üretmedi. Bunu göstermezsek kullanıcı
            # boş bir sohbet balonuyla baş başa kalır (gerçekte gözlemlenen
            # davranış) — burada da yalnızca `warning`.
            logger.warning("Empty model answer: %s", exc)
            yield _sse({"type": "error", "detail": str(exc)})
            return
        except Exception:
            # Son çare güvenlik ağı: `generate_answer_stream` zaten bilinen
            # Foundry Local bağlantı sorunlarını (bkz. llm.py:_stream_and_strip)
            # FoundryNotAvailable'a çeviriyor, ama beklenmedik bir hata
            # (ör. Foundry Local'in kendisi 500 dönerse) burayı da yakalamazsak
            # istemci hiçbir `error`/`done` olayı almadan bağlantının aniden
            # kesildiğini görür — ekranda sonsuza kadar "yazıyor..." kalır.
            # Bu yüzden ne olursa olsun bir `error` olayıyla akışı düzgün
            # bitiriyoruz; ayrıntıyı da sunucu loguna yazıyoruz.
            logger.exception("/ask/stream sırasında beklenmeyen hata")
            yield _sse(
                {
                    "type": "error",
                    "detail": "Cevap üretilirken beklenmeyen bir hata oluştu. "
                    "Backend'in çalıştığı terminale bakın.",
                }
            )
            return
        yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Ters proxy'lerin (ör. nginx) akışı tamponlayıp tek seferde
            # göndermesini engeller; token'ların gerçekten parça parça
            # ulaşması için gerekli.
            "X-Accel-Buffering": "no",
        },
    )
