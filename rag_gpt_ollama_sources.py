import streamlit as st
import faiss
import numpy as np
import subprocess
import os
import time
import requests
from sentence_transformers import SentenceTransformer

# ======================================================
# CONFIGURATION
# ======================================================
EMBED_MODEL = "all-MiniLM-L6-v2"
VECTOR_DB = "rag_index.faiss"
DOC_STORE = "rag_docs.npy"
OLLAMA_MODEL = "phi3:mini"  # or 'llama3' if available
SERPER_API_KEY = ""

# ======================================================
# STREAMLIT PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Hybrid RAG Q&A", page_icon="🧠", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
<style>
body { background-color: #0E1117; color: #FAFAFA; }
.header {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(to right, #3F8CFF, #6FE8C3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subheader { text-align: center; color: #B8BFC9; margin-bottom: 1.5rem; }
.answer-box {
    background: linear-gradient(145deg, #1E232B, #1A1D24);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #3A3F4B;
    margin-top: 10px;
}
.source-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid #333;
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 14px;
}
.footer {
    text-align: center;
    color: #6F7785;
    margin-top: 2rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# CACHE RESOURCES
# ======================================================
@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def load_faiss():
    if os.path.exists(VECTOR_DB) and os.path.exists(DOC_STORE):
        try:
            index = faiss.read_index(VECTOR_DB)
            docs = np.load(DOC_STORE, allow_pickle=True).tolist()
        except:
            index, docs = faiss.IndexFlatL2(384), []
    else:
        index, docs = faiss.IndexFlatL2(384), []
    return index, docs

@st.cache_data(ttl=60)
def is_connected():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except:
        return False

embedder = get_embedder()
index, doc_store = load_faiss()

# ======================================================
# RAG CORE FUNCTIONS
# ======================================================
def fetch_web_results(query, num_results=5):
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    data = {"q": query, "num": num_results, "hl": "en"}
    try:
        res = requests.post("https://google.serper.dev/search", headers=headers, json=data, timeout=10)
        results = res.json().get("organic", [])
        docs = []
        for r in results:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            text = f"Title: {title}\nSnippet: {snippet}"
            docs.append({"text": text.strip(), "source": link})
        return docs
    except Exception as e:
        st.warning(f"⚠️ Web fetch error: {e}")
        return []

def add_to_index(text, source):
    vec = embedder.encode([text]).astype("float32")
    index.add(vec)
    doc_store.append({"text": text, "source": source})
    faiss.write_index(index, VECTOR_DB)
    np.save(DOC_STORE, np.array(doc_store, dtype=object))

def generate_with_ollama(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="ignore"
        )
        return result.stdout.strip()
    except FileNotFoundError:
        return "❌ Ollama not found. Please install it from https://ollama.ai/"
    except Exception as e:
        return f"❌ Ollama Error: {e}"

def offline_rag(query, top_k=3):
    if not doc_store or index.ntotal == 0:
        return "⚠️ No offline data. Try online mode once to build cache.", []
    vec = embedder.encode([query]).astype("float32")
    _, idx = index.search(vec, top_k)
    docs = [doc_store[i] for i in idx[0] if i < len(doc_store)]
    if not docs:
        return "⚠️ No relevant offline information found.", []
    context = "\n\n".join(d["text"] for d in docs)
    sources = [d["source"] for d in docs if d["source"]]
    prompt = f"""
Answer factually using the given CONTEXT. 
If unsure, say "Information not found locally."

CONTEXT:
{context}

QUESTION:
{query}

Answer clearly:
"""
    return generate_with_ollama(prompt), sources

def online_rag(query):
    docs = fetch_web_results(query)
    if not docs:
        return "⚠️ No relevant web info found.", []
    for d in docs:
        add_to_index(d["text"], d["source"])
    context = "\n\n".join(d["text"] for d in docs)
    sources = [d["source"] for d in docs if d["source"]]
    prompt = f"""
Answer factually using this recent web search context.

CONTEXT:
{context}

QUESTION:
{query}

Answer clearly:
"""
    return generate_with_ollama(prompt), sources

# ======================================================
# ANIMATED TYPING EFFECT
# ======================================================
def animated_typing(text, speed=0.015):
    container = st.empty()
    output = ""
    for c in text:
        output += c
        container.markdown(f"<div class='answer-box'>{output}▌</div>", unsafe_allow_html=True)
        time.sleep(speed)
    container.markdown(f"<div class='answer-box'>{output}</div>", unsafe_allow_html=True)

# ======================================================
# SIDEBAR (HISTORY + MODE)
# ======================================================
if "history" not in st.session_state:
    st.session_state["history"] = []

with st.sidebar:
    st.header("🧾 Chat History")
    if st.session_state["history"]:
        if st.button("🧹 Clear History"):
            st.session_state["history"].clear()
            st.experimental_rerun()
        for item in reversed(st.session_state["history"]):
            with st.expander(item["question"]):
                st.caption(item["answer"][:100] + "...")
    else:
        st.info("No chat history yet.")

    st.markdown("---")
    online_status = is_connected()
    if online_status:
        st.success("🌍 Internet Connected")
        mode = st.radio("Select Mode:", ["🌍 Online (Serper + Ollama)", "💻 Offline (FAISS + Ollama)"], index=0)
    else:
        st.warning("⚠️ No Internet Connection — Offline Mode Only")
        mode = "💻 Offline (FAISS + Ollama)"
    st.caption(f"📂 {index.ntotal} docs cached locally.")
    st.markdown("---")

# ======================================================
# MAIN INTERFACE
# ======================================================
st.markdown("<div class='header'>🧠 Enhancing Question Answering with GPT + RAG</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Hybrid AI System | 🌍 Online (Serper) + 💻 Offline (FAISS) | Powered by Ollama</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Use Streamlit form for stable submission
with st.form(key="qa_form"):
    query = st.text_input("Enter your question:", placeholder="e.g., What is Artificial Intelligence?")
    submit = st.form_submit_button("🚀 Run Query")

if submit and query.strip():
    st.session_state["history"].append({"question": query, "answer": "..."})
    with st.spinner("⚙️ Generating intelligent answer..."):
        if "Online" in mode:
            answer, sources = online_rag(query)
        else:
            answer, sources = offline_rag(query)
    st.session_state["history"][-1]["answer"] = answer

    st.markdown("### 🧠 Assistant:")
    animated_typing(answer)

    if sources:
        st.markdown("### 📚 Sources:")
        for s in set(sources):
            st.markdown(f"<div class='source-card'>🔗 <a href='{s}' target='_blank'>{s}</a></div>", unsafe_allow_html=True)

elif not st.session_state["history"]:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.05);padding:2rem;border-radius:12px;text-align:center;'>
        <h3>👋 Welcome to the Hybrid RAG Q&A System!</h3>
        <p>Ask a question and click <b>Run Query</b> to get answers.<br>
        Works both <b>online</b> and <b>offline</b>.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='footer'>Developed by <b>Sagar Rajgiri</b> | VIT Vellore | NLP Project 2025</div>", unsafe_allow_html=True)
