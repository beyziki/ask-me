"""summary.py'deki saf (network/Foundry gerektirmeyen) yardımcılar için
birim testleri: map-reduce gruplama ve prompt oluşturma.

Özet, dokümanın TAMAMINI modelden geçirdiği için uzun dosyalarda tek istekte
yapılamıyor; doküman ardışık gruplara bölünüp önce her grup ayrı özetleniyor
(map), sonra ara özetler birleştiriliyor (reduce). Gruplama mantığı bu akışın
kalbi — sıra bozulursa özet de bozulur.
"""
import os
import tempfile
from pathlib import Path

_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="ask-me-test-summary-"))
os.environ.setdefault("DATA_DIR", str(_TEST_DATA_DIR))
os.environ.setdefault("UPLOADS_DIR", str(_TEST_DATA_DIR / "uploads"))
os.environ.setdefault("VECTORSTORE_DIR", str(_TEST_DATA_DIR / "vectorstore"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TEST_DATA_DIR / 'test.db'}")

import pytest  # noqa: E402

from backend.app.services import summary as summary_service  # noqa: E402
from backend.app.services.llm import (  # noqa: E402
    GpuContextLost,
    is_capacity_error,
    is_fatal_gpu_error,
    raise_if_gpu_context_lost,
)
from backend.app.services.summary import (  # noqa: E402
    SYSTEM_PROMPT_PARTIAL_TR,
    _split_for_retry,
    _split_text,
    _summarize_with_fallback,
    SYSTEM_PROMPT_SUMMARY_EN,
    SYSTEM_PROMPT_SUMMARY_TR,
    _build_messages,
    _system_prompt,
    group_chunks,
)


# --- Gruplama -------------------------------------------------------------


def test_group_chunks_keeps_everything_in_one_group_when_short():
    # Kısa dokümanlarda map adımı hiç çalışmamalı: tek grup -> tek LLM
    # çağrısı, gereksiz ikinci tur maliyeti yok.
    chunks = ["a" * 100, "b" * 100]
    assert group_chunks(chunks, max_chars=1000) == [chunks]


def test_group_chunks_splits_when_exceeding_limit():
    chunks = ["a" * 400, "b" * 400, "c" * 400]
    groups = group_chunks(chunks, max_chars=900)
    assert groups == [[chunks[0], chunks[1]], [chunks[2]]]


def test_group_chunks_preserves_document_order():
    # Sıra korunmalı: özet için parçaların doküman sırasında kalması şart,
    # aksi halde "önce sonuç, sonra tanım" gibi bozuk bir anlatım çıkar.
    chunks = [f"parça-{i}" for i in range(10)]
    flattened = [chunk for group in group_chunks(chunks, max_chars=20) for chunk in group]
    assert flattened == chunks


def test_group_chunks_loses_nothing():
    chunks = [f"{i}" * 50 for i in range(7)]
    groups = group_chunks(chunks, max_chars=120)
    assert sum(len(g) for g in groups) == len(chunks)


def test_group_chunks_keeps_oversized_chunk_as_its_own_group():
    # Tek bir parça sınırdan büyükse bölmüyoruz (chunking zaten ingestion'da
    # yapıldı; burada tekrar bölmek cümle ortasından kesme riski taşır).
    big = "x" * 5000
    groups = group_chunks(["kısa", big, "kısa2"], max_chars=1000)
    assert [big] in groups
    assert sum(len(g) for g in groups) == 3


def test_group_chunks_handles_empty_input():
    assert group_chunks([], max_chars=1000) == []


# --- Prompt ---------------------------------------------------------------


def test_system_prompt_differs_between_partial_and_final():
    # Ara (map) özetler kullanıcıya gösterilmiyor: oradan biçimlendirme değil,
    # bilgi yoğunluğu isteniyor.
    assert _system_prompt("tr", partial=True) == SYSTEM_PROMPT_PARTIAL_TR
    assert _system_prompt("tr", partial=False) == SYSTEM_PROMPT_SUMMARY_TR
    assert _system_prompt("tr", partial=True) != _system_prompt("tr", partial=False)


def test_system_prompt_respects_language():
    assert _system_prompt("en", partial=False) == SYSTEM_PROMPT_SUMMARY_EN
    assert _system_prompt("tr", partial=False) != _system_prompt("en", partial=False)


def test_build_messages_includes_all_given_texts():
    messages = _build_messages(["birinci parça", "ikinci parça"], "tr", partial=False)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "birinci parça" in messages[1]["content"]
    assert "ikinci parça" in messages[1]["content"]


def test_final_summary_prompt_asks_for_markdown():
    # Özet arayüzde markdown olarak render ediliyor (bkz.
    # frontend-web/src/components/Markdown.tsx).
    assert "markdown" in SYSTEM_PROMPT_SUMMARY_TR.lower()
    assert "markdown" in SYSTEM_PROMPT_SUMMARY_EN.lower()


# --- GPU kapasite hatasına karşı bölerek yeniden deneme --------------------
# Gözlemlenen: 12B model + 8GB VRAM'de uzun context, Foundry Local'den
# "CUDA error in CudaMallocArray ... illegal memory access" (HTTP 500)
# döndürüyor. Kalıcı bir hata değil — daha küçük context'le çalışıyor.


class _CapacityError(Exception):
    pass


_ILLEGAL_ACCESS_MESSAGE = (
    "Failed to handle OpenAI completion: CUDA error in CudaMallocArray "
    "- an illegal memory access was encountered"
)


def test_illegal_memory_access_is_fatal_not_retryable():
    # KRİTİK AYRIM: "illegal memory access" CUDA bağlamını bozuyor; ondan
    # sonra HER istek (boyutu ne olursa olsun) başarısız oluyor. Gerçek
    # gözlem: 3145 -> 1562 -> 770 -> 385 karakter, dördü de aynı hatayı
    # verdi. Bunu "kapasite" sayıp bölerek tekrar denemek, ölü bir GPU'ya
    # boşuna vurmak demek.
    exc = _CapacityError(_ILLEGAL_ACCESS_MESSAGE)
    assert is_fatal_gpu_error(exc)
    assert not is_capacity_error(exc)


def test_out_of_memory_is_retryable_capacity_error():
    exc = _CapacityError("CUDA out of memory")
    assert is_capacity_error(exc)
    assert not is_fatal_gpu_error(exc)


def test_capacity_check_ignores_unrelated_errors():
    assert not is_capacity_error(_CapacityError("Doküman bulunamadı"))
    assert not is_fatal_gpu_error(_CapacityError("invalid json"))


def test_raise_if_gpu_context_lost_gives_actionable_message():
    with pytest.raises(GpuContextLost) as info:
        raise_if_gpu_context_lost(_CapacityError(_ILLEGAL_ACCESS_MESSAGE))
    message = str(info.value)
    # Kullanıcı ne yapacağını bilmeli: servisi yeniden başlat / küçük model.
    assert "foundry-doctor" in message
    assert "FOUNDRY_MODEL_ALIAS" in message


def test_raise_if_gpu_context_lost_is_noop_for_other_errors():
    raise_if_gpu_context_lost(_CapacityError("CUDA out of memory"))  # hata fırlatmamalı


def test_summarize_with_fallback_does_not_retry_fatal_gpu_error(monkeypatch):
    # Bölerek tekrar deneme YAPILMAMALI; tek çağrıdan sonra anlaşılır bir
    # hataya çevrilmeli.
    calls = []

    def fails_fatally(texts, language, partial, max_tokens):
        calls.append(len(texts))
        raise _CapacityError(_ILLEGAL_ACCESS_MESSAGE)

    monkeypatch.setattr(summary_service, "_summarize_once", fails_fatally)

    with pytest.raises(GpuContextLost):
        _summarize_with_fallback(["a", "b", "c", "d"], "tr", partial=True, max_tokens=500)
    assert calls == [4]


def test_summarize_with_fallback_splits_on_capacity_error(monkeypatch):
    # İlk (tam) çağrı kapasiteye çarpsın; bölünmüş çağrılar geçsin.
    seen: list[int] = []

    def fake_summarize_once(texts, language, partial, max_tokens):
        seen.append(len(texts))
        if len(texts) > 2:
            raise _CapacityError("CUDA out of memory")
        return f"özet({len(texts)})"

    monkeypatch.setattr(summary_service, "_summarize_once", fake_summarize_once)

    result = _summarize_with_fallback(["a", "b", "c", "d"], "tr", partial=True, max_tokens=500)

    assert seen[0] == 4  # önce tamamı denendi
    assert 2 in seen  # sonra ikiye bölündü
    assert "özet(2)" in result


def test_summarize_with_fallback_reraises_when_nothing_left_to_split(monkeypatch):
    # Tek bir parça bile sığmıyorsa bölecek bir şey kalmıyor; hata gizlenmemeli
    # ki kullanıcı gerçek nedeni görsün.
    def always_fails(texts, language, partial, max_tokens):
        raise _CapacityError("CUDA out of memory")

    monkeypatch.setattr(summary_service, "_summarize_once", always_fails)

    with pytest.raises(_CapacityError):
        _summarize_with_fallback(["tek parça"], "tr", partial=True, max_tokens=500)


def test_summarize_with_fallback_does_not_retry_unrelated_errors(monkeypatch):
    # Kapasiteyle ilgisi olmayan hatalarda bölerek tekrar denemek anlamsız —
    # aynı hata tekrar edecektir, sadece zaman kaybı olur.
    calls = []

    def fails_differently(texts, language, partial, max_tokens):
        calls.append(len(texts))
        raise ValueError("beklenmedik bir şey")

    monkeypatch.setattr(summary_service, "_summarize_once", fails_differently)

    with pytest.raises(ValueError):
        _summarize_with_fallback(["a", "b", "c", "d"], "tr", partial=True, max_tokens=500)
    assert calls == [4]  # tek deneme, bölme yok


def test_split_text_splits_at_word_boundary():
    text = "kelime " * 200  # ~1400 karakter
    halves = _split_text(text)
    assert halves is not None
    # Kelime ortasından kesilmemeli.
    assert not halves[0].endswith("keli")
    assert "".join(halves).replace(" ", "") == text.replace(" ", "")


def test_split_text_returns_none_when_too_short_to_split():
    assert _split_text("kısa metin") is None


def test_split_for_retry_splits_single_large_chunk():
    # ASIL ARIZA: `chunk_size` küçültülmeden önce yüklenmiş dosyaların
    # parçaları çok büyük (~3000+ karakter), yani bir grup TEK parçadan
    # oluşuyor. Liste bölünemediği için fallback hiç devreye girmiyordu.
    big = "kelime " * 500
    pieces = _split_for_retry([big])
    assert pieces is not None
    left, right = pieces
    assert len(left) == 1 and len(right) == 1
    assert len(left[0]) < len(big) and len(right[0]) < len(big)


def test_split_for_retry_splits_list_when_multiple_chunks():
    left, right = _split_for_retry(["a", "b", "c", "d"])
    assert left == ["a", "b"]
    assert right == ["c", "d"]


def test_split_for_retry_gives_up_on_tiny_single_chunk():
    assert _split_for_retry(["çok kısa"]) is None
    assert _split_for_retry([]) is None


def test_summarize_with_fallback_splits_single_oversized_chunk(monkeypatch):
    # Uçtan uca: tek ve büyük bir parça kapasiteye çarpınca metin bölünüp
    # yeniden denenmeli (eskiden burada hata yükseliyordu).
    big = "kelime " * 500
    seen: list[int] = []

    def fake_summarize_once(texts, language, partial, max_tokens):
        total = sum(len(t) for t in texts)
        seen.append(total)
        if total > len(big) / 2 + 10:
            raise _CapacityError("CUDA out of memory")
        return "özet"

    monkeypatch.setattr(summary_service, "_summarize_once", fake_summarize_once)

    result = _summarize_with_fallback([big], "tr", partial=True, max_tokens=500)

    assert "özet" in result
    assert len(seen) > 1  # ilk deneme çarptı, bölünmüş denemeler geçti
