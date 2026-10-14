"""
Streamlit web app for the RAG-Based Intelligent Document Assistant.

Run:   streamlit run app.py
"""
import os

import streamlit as st

st.set_page_config(page_title="Intelligent Document Assistant", page_icon="📚", layout="wide")

# On Streamlit Community Cloud the API keys come from "Secrets" instead of .env
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

from doc_assistant.config import DATA_DIR, PERSIST_DIR, configure_llamaindex, get_llm, provider_label  # noqa: E402
from doc_assistant.indexer import add_documents, list_documents, load_files, load_or_build_index, load_web_page  # noqa: E402
from doc_assistant.rag_chain import DocumentAssistant  # noqa: E402

CONF_BADGE = {"high": "🟢 high", "medium": "🟡 medium", "low": "🔴 low"}


@st.cache_resource(show_spinner="Loading embedding model and document index (first run takes ~1 minute)…")
def load_index():
    configure_llamaindex()
    return load_or_build_index()


@st.cache_resource(show_spinner=False)
def load_llm():
    return get_llm()


def get_assistant(index):
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = DocumentAssistant(index, load_llm())
    return st.session_state["assistant"]


def show_sources(sources):
    with st.expander(f"📎 Sources ({len(sources)} chunks)"):
        for s in sources:
            page = f" · page {s['page']}" if s.get("page") else ""
            st.markdown(f"**[{s['n']}] {s['file_name']}{page}** — similarity `{s['score']}`")
            st.caption(s["text"][:700] + ("…" if len(s["text"]) > 700 else ""))


try:
    index = load_index()
    assistant = get_assistant(index)
except Exception as e:
    st.error(f"Startup error: {e}\n\nCheck your `.env` file (see README).")
    st.stop()

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.title("📚 Document Assistant")
    st.caption(f"LLM (LangChain): `{provider_label()}`  \nIndex (LlamaIndex): `{PERSIST_DIR.name}/`")

    st.subheader("1. Add documents")
    uploads = st.file_uploader("PDF, TXT or MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploads and st.button("Index uploaded files", type="primary"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        existing = set(list_documents(index))
        paths = []
        for f in uploads:
            if f.name in existing:
                st.warning(f"{f.name} is already indexed - skipped.")
                continue
            p = DATA_DIR / f.name
            p.write_bytes(f.getbuffer())
            paths.append(p)
        if paths:
            with st.spinner("Chunking + embedding…"):
                add_documents(index, load_files(paths))
            st.success(f"Indexed {len(paths)} file(s).")

    url = st.text_input("…or a web page URL")
    if url and st.button("Index web page"):
        try:
            with st.spinner("Fetching page…"):
                add_documents(index, load_web_page(url))
            st.success("Web page indexed.")
        except Exception as e:
            st.error(f"Could not load page: {e}")

    st.subheader("2. Choose documents")
    all_docs = list_documents(index)
    selected = st.multiselect("Search in (empty = all documents)", all_docs)

    st.subheader("3. Retrieval settings")
    assistant.top_k = st.slider("Chunks to retrieve (top-k)", 2, 8, 4)
    assistant.similarity_cutoff = st.slider("Similarity cutoff", 0.0, 0.8, 0.25, 0.05)

    if st.button("🧹 Clear chat memory"):
        assistant.reset_memory()
        st.session_state["messages"] = []

if not all_docs:
    st.info("No documents yet. Upload a PDF/TXT/MD file or add a URL in the sidebar.")
    st.stop()

tab_chat, tab_sum, tab_cmp = st.tabs(["💬 Ask questions", "📝 Summarize a document", "⚖️ Compare documents"])

# ---------------------------------------------------------------- Chat
with tab_chat:
    st.caption(f"{len(all_docs)} document(s) indexed: " + ", ".join(all_docs))
    st.session_state.setdefault("messages", [])
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                show_sources(m["sources"])

    if question := st.chat_input("Ask a question about your documents…"):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving with LlamaIndex → answering with LangChain…"):
                try:
                    res = assistant.ask(question, file_names=selected or None)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()
            text = res["answer"] + f"\n\n*Confidence: {CONF_BADGE.get(res['confidence'], res['confidence'])}*"
            if res["standalone_question"] != question:
                text += f"  \n*Interpreted as: “{res['standalone_question']}”*"
            if res.get("follow_up_questions"):
                text += "\n\n**You could also ask:**\n" + "\n".join(f"- {q}" for q in res["follow_up_questions"])
            st.markdown(text)
            show_sources(res["sources"])
        st.session_state["messages"].append({"role": "assistant", "content": text, "sources": res["sources"]})

# ---------------------------------------------------------------- Summarize
with tab_sum:
    doc = st.selectbox("Document", all_docs, key="sum_doc")
    if st.button("Summarize", type="primary"):
        with st.spinner("Map-reduce summarisation…"):
            st.session_state["summary"] = (doc, assistant.summarize(doc))
    if st.session_state.get("summary"):
        name, summary = st.session_state["summary"]
        st.subheader(name)
        st.markdown(summary)

# ---------------------------------------------------------------- Compare
with tab_cmp:
    docs_cmp = st.multiselect("Pick 2 or more documents", all_docs, default=all_docs[:2], key="cmp_docs")
    q_cmp = st.text_input("Question", value="What problem does this document address and what is the main idea?")
    if st.button("Compare", type="primary", disabled=len(docs_cmp) < 2):
        with st.spinner("Retrieving from each document…"):
            res = assistant.compare(q_cmp, docs_cmp)
        st.markdown(res["answer"])
        show_sources(res["sources"])
