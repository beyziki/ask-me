"""Ask Me? - Streamlit arayüzü (MVP).

Backend'in ayrı bir terminalde çalışıyor olması gerekir:
    uvicorn backend.app.main:app --reload
"""
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Ask Me? - Offline AI Study Assistant", page_icon="📚")

if "user" not in st.session_state:
    st.session_state.user = None


def auth_screen():
    st.title("Ask Me? 📚")
    st.caption("Offline AI Study Assistant")

    tab_login, tab_register = st.tabs(["Giriş Yap", "Kayıt Ol"])

    with tab_login:
        username = st.text_input("Kullanıcı adı", key="login_user")
        password = st.text_input("Şifre", type="password", key="login_pass")
        if st.button("Giriş Yap"):
            resp = requests.post(
                f"{API_BASE}/users/login",
                json={"username": username, "password": password},
            )
            if resp.ok:
                st.session_state.user = resp.json()
                st.rerun()
            else:
                st.error("Kullanıcı adı veya şifre hatalı")

    with tab_register:
        username = st.text_input("Kullanıcı adı", key="reg_user")
        password = st.text_input("Şifre", type="password", key="reg_pass")
        lang = st.selectbox("Dil tercihi", ["tr", "en"], key="reg_lang")
        if st.button("Kayıt Ol"):
            resp = requests.post(
                f"{API_BASE}/users/register",
                json={"username": username, "password": password, "preferred_language": lang},
            )
            if resp.ok:
                st.success("Kayıt başarılı, şimdi giriş yapabilirsiniz.")
            else:
                st.error(resp.json().get("detail", "Kayıt başarısız"))


def main_app():
    user = st.session_state.user
    headers = {"X-User-Id": str(user["id"])}

    st.sidebar.write(f"👤 {user['username']}")
    if st.sidebar.button("Çıkış Yap"):
        st.session_state.user = None
        st.rerun()

    page = st.sidebar.radio("Menü", ["Dosya Yükle", "Soru Sor", "Quiz", "Kod Analizi"])

    if page == "Dosya Yükle":
        st.header("📄 Dosya Yükle")
        uploaded = st.file_uploader("PDF, Markdown veya kod dosyası seçin")
        if uploaded and st.button("Yükle"):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            resp = requests.post(f"{API_BASE}/documents/upload", headers=headers, files=files)
            if resp.ok:
                st.success(f"{uploaded.name} başarıyla yüklendi.")
            else:
                st.error(resp.text)

        st.subheader("Yüklenen Dosyalar")
        resp = requests.get(f"{API_BASE}/documents", headers=headers)
        if resp.ok:
            for doc in resp.json():
                st.write(f"- {doc['filename']} ({doc['file_type']})")

    elif page == "Soru Sor":
        st.header("💬 Soru Sor")
        question = st.text_area("Sorunuz")
        if st.button("Gönder") and question:
            resp = requests.post(f"{API_BASE}/ask", headers=headers, json={"question": question})
            if resp.ok:
                data = resp.json()
                st.markdown(data["answer"])
                with st.expander("Kaynaklar"):
                    for src in data["sources"]:
                        st.write(f"**{src['filename']}** (parça {src['chunk_index']})")
                        st.caption(src["snippet"])
            else:
                st.error(resp.text)

    elif page == "Quiz":
        st.header("📝 Quiz Oluştur")
        resp = requests.get(f"{API_BASE}/documents", headers=headers)
        docs = resp.json() if resp.ok else []
        doc_map = {d["filename"]: d["id"] for d in docs}
        selected = st.selectbox("Doküman seçin", list(doc_map.keys()) if doc_map else ["-"])
        num_q = st.slider("Soru sayısı", 1, 10, 5)
        if st.button("Quiz Oluştur") and doc_map:
            resp = requests.post(
                f"{API_BASE}/quiz",
                headers=headers,
                json={"document_id": doc_map[selected], "num_questions": num_q},
            )
            if resp.ok:
                quiz = resp.json()
                for i, q in enumerate(quiz["questions"], 1):
                    st.write(f"**{i}. {q['question']}**")
                    if q["options"]:
                        st.write(", ".join(q["options"]))
                    with st.expander("Cevabı göster"):
                        st.write(q["answer"])
            else:
                st.error(resp.text)

    elif page == "Kod Analizi":
        st.header("🧑‍💻 Kod Analizi")
        resp = requests.get(f"{API_BASE}/documents", headers=headers)
        docs = [d for d in (resp.json() if resp.ok else []) if d["file_type"] == "code"]
        doc_map = {d["filename"]: d["id"] for d in docs}
        if not doc_map:
            st.info("Henüz yüklenmiş kod dosyası yok.")
        else:
            selected = st.selectbox("Kod dosyası seçin", list(doc_map.keys()))
            if st.button("Açıkla"):
                resp = requests.post(
                    f"{API_BASE}/code/explain",
                    headers=headers,
                    json={"document_id": doc_map[selected]},
                )
                if resp.ok:
                    st.markdown(resp.json()["explanation"])
                else:
                    st.error(resp.text)


if st.session_state.user is None:
    auth_screen()
else:
    main_app()
