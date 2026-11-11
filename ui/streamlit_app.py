"""Streamlit demo interface: talks to the FastAPI backend."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄")
st.title("📄 RAG Document Assistant")
st.caption("Ask questions about your PDFs. Every answer cites its sources.")

with st.sidebar:
    st.header("1. Add a document")
    uploaded = st.file_uploader("PDF file", type=["pdf"])
    if uploaded and st.button("Index this PDF"):
        with st.spinner("Reading, chunking and embedding..."):
            r = requests.post(f"{API_URL}/upload", files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")})
        if r.ok:
            st.success(f"{r.json()['chunks_indexed']} chunks indexed")
        else:
            st.error(r.json().get("detail", r.text))
    top_k = st.slider("Passages retrieved (top-k)", 1, 10, 5)

st.header("2. Ask a question")
question = st.text_input("Your question")
if st.button("Ask", type="primary") and question:
    with st.spinner("Searching and generating..."):
        r = requests.post(f"{API_URL}/ask", json={"question": question, "top_k": top_k})
    if not r.ok:
        st.error(r.text)
    else:
        data = r.json()
        st.markdown(data["answer"])
        st.subheader("Sources")
        for i, s in enumerate(data["sources"], start=1):
            with st.expander(f"[{i}] {s['source']} – page {s['page']} (similarity {s['score']:.2f})"):
                st.write(s["text"])
