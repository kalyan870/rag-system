import streamlit as st
import httpx
from pathlib import Path

API_URL = st.secrets.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG System", page_icon="📄", layout="wide")

st.title("📄 Production RAG System")
st.markdown("Upload documents and ask questions with AI-powered retrieval & citation generation.")


def api_client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=120)


with st.sidebar:
    st.header("⚙️ Settings")
    api_url_input = st.text_input("API URL", value=API_URL)
    if api_url_input != API_URL:
        API_URL = api_url_input
        st.rerun()

    st.divider()
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Upload & Ingest", type="primary", use_container_width=True):
            with st.spinner("Processing documents..."):
                for f in uploaded_files:
                    files = {"file": (f.name, f.read(), f.type)}
                    try:
                        resp = api_client().post("/upload", files=files)
                        if resp.status_code == 200:
                            st.success(f"✅ {f.name} ingested")
                        else:
                            st.error(f"❌ {f.name}: {resp.json().get('detail', 'Error')}")
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")

    st.divider()
    if st.button("🔄 Refresh Documents", use_container_width=True):
        st.rerun()

tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📚 Documents", "📊 Evaluate"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_area("Ask a question about your documents:", height=100, placeholder="What are the key findings in the report?")
    with col2:
        st.write("")
        st.write("")
        ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)

    if ask_btn and query.strip():
        with st.spinner("Searching documents and generating answer..."):
            try:
                resp = api_client().post("/query", data={"query": query})
                if resp.status_code == 200:
                    result = resp.json()

                    st.subheader("💡 Answer")
                    st.markdown(result["answer"])

                    with st.expander("📎 View Citations & Context", expanded=True):
                        for i, ctx in enumerate(result.get("citations", [])):
                            score = ctx.get("relevance", 0)
                            st.markdown(f"**Context {i+1}** — *{ctx['filename']}* (relevance: {score:.3f})")
                            st.text(ctx["text"][:500] + ("..." if len(ctx["text"]) > 500 else ""))
                            st.divider()
                else:
                    st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Connection error: {e}")

with tab2:
    st.subheader("📚 Ingested Documents")
    try:
        resp = api_client().get("/documents")
        if resp.status_code == 200:
            docs = resp.json().get("documents", [])
            if docs:
                for d in docs:
                    with st.container(border=True):
                        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                        c1.write(f"**{d['filename']}**")
                        c2.write(f"Chunks: {d['chunk_count']}")
                        c3.write(d["uploaded_at"][:10])
                        if c4.button("🗑️", key=f"del_{d['doc_id']}"):
                            api_client().delete(f"/documents/{d['doc_id']}")
                            st.rerun()
            else:
                st.info("No documents ingested yet. Upload files in the sidebar.")
    except Exception as e:
        st.error(f"Connection error: {e}")

with tab3:
    st.subheader("📊 Evaluation")
    st.markdown("Compare generated answers against reference answers.")
    eval_query = st.text_area("Question", key="eval_q", placeholder="Enter the question")
    eval_ref = st.text_area("Reference Answer", key="eval_ref", placeholder="Enter the ideal answer")
    eval_gen = st.text_area("Generated Answer", key="eval_gen", placeholder="Enter the answer to evaluate")

    if st.button("Evaluate", type="primary") and eval_query and eval_ref and eval_gen:
        with st.spinner("Computing metrics..."):
            try:
                resp = api_client().post(
                    "/evaluate",
                    data={
                        "query": eval_query,
                        "reference_answer": eval_ref,
                        "generated_answer": eval_gen,
                    },
                )
                if resp.status_code == 200:
                    metrics = resp.json()
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("ROUGE-1", f"{metrics.get('rouge1', 0):.3f}")
                    col2.metric("ROUGE-2", f"{metrics.get('rouge2', 0):.3f}")
                    col3.metric("ROUGE-L", f"{metrics.get('rougeL', 0):.3f}")
                    col4.metric("Faithfulness", f"{metrics.get('faithfulness', 0):.3f}")
                    if "bert_score_f1" in metrics:
                        st.metric("BERTScore F1", f"{metrics['bert_score_f1']:.3f}")
                else:
                    st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Connection error: {e}")

st.sidebar.divider()
st.sidebar.markdown("**🔗 API Status**")
try:
    r = api_client().get("/health")
    if r.status_code == 200:
        st.sidebar.success("🟢 Backend connected")
    else:
        st.sidebar.error("🔴 Backend error")
except:
    st.sidebar.warning("🟡 Cannot reach backend")
