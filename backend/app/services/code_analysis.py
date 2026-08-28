"""Kod dosyalarının analiz edilip açıklanması (Foundry Local üzerinden)."""
from __future__ import annotations

from backend.app.services.llm import _get_manager, _get_model_id, strip_think

CODE_ANALYSIS_PROMPT_TR = (
    "Aşağıdaki kod dosyasını bir bilgisayar mühendisliği öğrencisine anlayacağı "
    "şekilde açıkla: genel amacı, ana fonksiyon/sınıfları, önemli algoritma veya "
    "veri yapılarını ve varsa dikkat edilmesi gereken noktaları (karmaşıklık, "
    "hata durumları vb.) belirt."
)

CODE_ANALYSIS_PROMPT_EN = (
    "Explain the following code file for a computer engineering student: its "
    "overall purpose, main functions/classes, key algorithms or data "
    "structures, and any notable points (complexity, edge cases, etc.)."
)


def explain_code(code_text: str, language: str = "tr") -> str:
    manager = _get_manager()
    model_id = _get_model_id()
    instruction = CODE_ANALYSIS_PROMPT_TR if language == "tr" else CODE_ANALYSIS_PROMPT_EN
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"{code_text}\n\n/no_think"},
    ]

    from openai import OpenAI

    client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.2,
        # bkz. llm.py generate_answer'daki max_tokens notu.
        max_tokens=900,
    )
    return strip_think(response.choices[0].message.content)
