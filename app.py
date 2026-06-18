"""
Agentic RAG — Streamlit
Agents: Ingestion → Rule Extraction → Router → QA
"""

import os, re, json
import streamlit as st
import numpy as np
import faiss

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURATION                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

BASE_URL = "https://api.openai.com/v1"         # swap provider here
MODEL    = "gpt-4o-mini"                        # swap model here
API_KEY  = os.environ.get("OPENAI_API_KEY", "") # set in Streamlit secrets

# Other providers (just change the two lines above):
# Groq:     BASE_URL = "https://api.groq.com/openai/v1"  | MODEL = "llama3-8b-8192"
# Together: BASE_URL = "https://api.together.xyz/v1"      | MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# Ollama:   BASE_URL = "http://localhost:11434/v1"         | MODEL = "mistral"
# vLLM:     BASE_URL = "http://localhost:8000/v1"          | MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# ══════════════════════════════════════════════════════════════════════════════

# ── LLM ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    from openai import OpenAI

    key = st.secrets.get("OPENAI_API_KEY", API_KEY).strip()

    

    return OpenAI(
        api_key=key,
        base_url=BASE_URL
    )

def llm(system, user, temperature=0.2):
    resp = get_client().chat.completions.create(
        model=MODEL, temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}]
    )
    return resp.choices[0].message.content.strip()

# ── Local embeddings ──────────────────────────────────────────────────────────
@st.cache_resource
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    vecs = get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype="float32")

# ── Document parsing ──────────────────────────────────────────────────────────
def parse_pdf(file):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(file)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        return f"[PDF error: {e}]"

def parse_docx(file):
    try:
        from docx import Document
        return "\n".join(p.text for p in Document(file).paragraphs)
    except Exception as e:
        return f"[DOCX error: {e}]"

def parse_document(uploaded_files):
    ext = os.path.splitext(uploaded_files.name)[-1].lower()
    if ext == ".pdf":  return parse_pdf(uploaded_files)
    if ext == ".docx": return parse_docx(uploaded_files)
    return uploaded_files.read().decode("utf-8", errors="ignore")

# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

# ── FAISS ─────────────────────────────────────────────────────────────────────
def build_index(chunks):
    embeddings = embed(chunks)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query, index, chunks, top_k=5):
    scores, idxs = index.search(embed([query]), top_k)
    return [
        {
            "source": chunks[idx]["source"],
            "chunk_id": chunks[idx]["chunk_id"],
            "text": chunks[idx]["text"],
            "score": float(score)
        }
        for score, idx in zip(scores[0], idxs[0])
        if idx >= 0
    ]

# ══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════════════════════

# Agent 1 — Ingestion
def ingestion_agent(uploaded_files):

    all_chunks = []

    documents = {}

    for file in uploaded_files:

        raw = parse_document(file)

        chunks = chunk_text(raw)

        documents[file.name] = chunks

        for idx, chunk in enumerate(chunks):

            all_chunks.append({
                "source": file.name,
                "chunk_id": idx,
                "text": chunk
            })

    texts = [c["text"] for c in all_chunks]

    embeddings = embed(texts)

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return {
        "chunks": all_chunks,
        "documents": documents,      # <-- keep actual document chunks
        "index": index,
        "summary": {},
        "rules": {},
        "memory": []
    }, f"Loaded {len(uploaded_files)} documents"

# Agent 2 — Rule Extraction
RULE_SYSTEM = """You are a Rule Extraction Agent.
Extract ALL explicit and implicit rules, constraints, policies, and guidelines from the document chunks.
Return ONLY a valid JSON array of strings. Each string = one self-contained rule in plain English.
No preamble, no markdown fences — just the raw JSON array."""

def rule_extraction_agent(doc):
    chunks = doc["chunks"]
    seen, top_chunks = set(), []
    for q in ["rules regulations policy", "must should shall not allowed", "requirements constraints"]:
        for r in retrieve(q, doc["index"], chunks, top_k=6):
            if r["chunk_id"] not in seen:
                seen.add(r["chunk_id"])
                top_chunks.append(r["text"])
    raw = llm(RULE_SYSTEM, "Document content:\n" + "\n\n---\n\n".join(top_chunks[:12]))
    try:
        rules = json.loads(re.sub(r"```(?:json)?|```", "", raw).strip())
        if isinstance(rules, list):
            return [str(r) for r in rules]
    except Exception:
        return [l.strip("•- ").strip() for l in raw.splitlines() if l.strip()]
    return []

# Agent 3 — Router
ROUTER_SYSTEM = """Classify the query into exactly one of:
- "rule_query"  — about rules, policies, constraints, allowed/not allowed
- "factual_qa"  — factual content question
- "summary" - about collective summary of the document, or overview or key findings
- "compare" - compare something amongst the multiple documents uploaded
Reply with ONLY the category string."""

def intelligent_router(query):

    q = query.lower()

    if any(k in q for k in [
        "summary",
        "summarize",
        "overview",
        "gist",
        "takeaway"
    ]):
        return "summary"

    if any(k in q for k in [
        "compare",
        "difference",
        "versus",
        "vs"
    ]):
        return "compare"

    result = llm(
        ROUTER_SYSTEM,
        query,
        temperature=0
    ).strip().lower()

    if result in [
        "summary",
        "compare",
        "rule_query",
        "factual_qa"
    ]:
        return result

    return "factual_qa"

# Agent 4 — QA
QA_SYSTEM = """You are a precise QA Agent. Answer strictly from the provided document context.

1. Base your answer ONLY on the given chunks and rules.
2. End every answer with a " Sources" section listing Chunk IDs and Rule numbers used.
3. If the answer is not in the context, say "Not found in document."

Format:
<your answer>

 Sources: Chunk #X, Chunk #Y | Rules: [Rule N]"""

def qa_agent(query, doc):
    retrieved = retrieve(query, doc["index"], doc["chunks"], top_k=5)
    if not retrieved:
        return "No relevant information found."

    top_score = retrieved[0]["score"]
    if top_score < 0.20:
        return (
            "The retrieved context has low confidence. "
            "Please specify the document or rephrase the question."
        )
        
    context = "\n\n".join(
        f"""
        Source: {r['source']}
        Chunk: {r['chunk_id']}
        Score: {r['score']:.2f}

        {r['text']}
        """
        for r in retrieved
    )
    rules = doc.get("rules", [])
    rules_text = ("\n\nDerived Rules:\n" + "\n".join(
        f"[Rule {i+1}] {r}" for i, r in enumerate(rules)
    )) if rules else ""
    memory = doc.get("memory", [])
    history = ""
    if memory:
        history = "\nConversation History:\n" + "".join(
            f"User: {t['q']}\nAssistant: {t['a']}\n\n" for t in memory[-4:]
        )
    answer = llm(QA_SYSTEM, f"{history}Context:\n{context}{rules_text}\n\nQuestion: {query}")
    doc["memory"].append({"q": query, "a": answer})
    return answer

# Agent 5 — Summary
SUMMARY_SYSTEM = """
You are a document summarization agent.

Create:
1. Executive Summary
2. Key Topics
3. Important Findings
4. Important Dates/Numbers

Keep it concise.
"""

def summary_agent(doc, target_doc=None):

    if target_doc:

        chunks = doc["documents"][target_doc]

        context = "\n\n".join(chunks[:20])

    else:

        context = "\n\n".join(
            c["text"]
            for c in doc["chunks"][:20]
        )

    return llm(
        SUMMARY_SYSTEM,
        context
    )

import re

def normalize(text):
    return re.sub(
        r'[^a-z0-9]',
        '',
        text.lower()
    )

def find_document(query, doc):

    q = normalize(query)

    for filename in doc["documents"]:

        clean_name = normalize(
            filename
            .replace(".pdf", "")
            .replace(".docx", "")
            .replace(".txt", "")
        )

        if clean_name in q:
            return filename

    return None

#Agent 6: Query Router 
def route_query(query, doc):

    q = query.lower()

    summary_keywords = [
        "summary",
        "summarize",
        "overview",
        "brief",
        "what is this document about"
    ]

    compare_keywords = [
        "compare",
        "difference",
        "differences",
        "versus",
        "vs"
    ]

    if any(k in q for k in compare_keywords):
        return "compare", compare_agent(query, doc)

    if any(k in q for k in summary_keywords):

        target_doc = find_document(query, doc)

        if target_doc:

            if target_doc not in doc["summary"]:

                doc["summary"][target_doc] = summary_agent(
                    doc,
                    target_doc
                )

            return (
                "summary",
                doc["summary"][target_doc]
            )

        if not target_doc:

            docs = "\n".join(
                f"- {d}"
                for d in doc["documents"].keys()
            )

            return (
                "summary",
                f"Multiple documents are uploaded.\n\nAvailable documents:\n{docs}\n\nPlease specify which document."
            )

    q_type = intelligent_router(query)

    if q_type == "rule_query":

        if not doc.get("rules"):
            doc["rules"] = rule_extraction_agent(doc)

        return "rule_query", qa_agent(query, doc)

    return "factual_qa", qa_agent(query, doc)

# Compare Agent
COMPARE_SYSTEM = """
Compare information across documents.
Highlight similarities, differences,
conflicts and overlaps.
"""
def compare_agent(query, doc):

    retrieved = retrieve(
        query,
        doc["index"],
        doc["chunks"],
        top_k=15
    )

    context = "\n\n".join(
        f"[{r['source']}]\n{r['text']}"
        for r in retrieved
    )

    return llm(
        COMPARE_SYSTEM,
        context + f"\n\nQuestion:{query}"
    )
# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")
st.title("🤖 Agentic RAG")
st.caption("Ingestion Agent → Rule Extraction Agent → Router Agent → QA Agent")

# Session state init
if "doc" not in st.session_state:
    st.session_state.doc = None

if "rules_text" not in st.session_state:
    st.session_state.rules_text = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 2])

with left:
    # Step 1 — Upload
    st.markdown("### 📁 Step 1 — Upload Document")
    uploaded_files = st.file_uploader(
        "PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    if st.button("⚡ Ingest Document", type="primary", use_container_width=True):

        if not uploaded_files:
            st.warning("Please upload a file first.")
        else:
            with st.spinner("Parsing and embedding..."):
                doc, msg = ingestion_agent(uploaded_files)
            if doc:
                st.session_state.doc = doc
                st.session_state.rules_text = ""

                st.session_state.chat_history = [
                    {
                        "role": "assistant",
                        "content":
                        f"Successfully loaded {len(doc['documents'])} document(s).\n\n"
                        f"Uploaded files:\n"
                        + "\n".join(
                            f"• {name}"
                            for name in doc["documents"].keys()
                        )
                        + "\n\nYou can now ask questions, request summaries, compare documents, or extract rules."
                    }
                ]

                st.success(msg)
            else:
                st.error(msg)
    #st.write("Secrets key prefix:", st.secrets.get("OPENAI_API_KEY", "")[:15])
    #st.write("Env key prefix:", os.environ.get("OPENAI_API_KEY", "")[:15])
    st.divider()

    # Step 2 — Extract Rules
    st.markdown("### 📋 Step 2 — Extract Rules")
    if st.button("🔍 Run Rule Extraction Agent", use_container_width=True):
        if not st.session_state.doc:
            st.warning("Ingest a document first.")
        else:
            with st.spinner("Extracting rules..."):
                rules = rule_extraction_agent(st.session_state.doc)
            st.session_state.doc["rules"] = rules
            st.session_state.rules_text = "\n".join(
                f"{i+1}. {r}" for i, r in enumerate(rules)
            ) if rules else "No rules found."
            st.success(f"Extracted **{len(rules)} rules**.")

    if st.session_state.rules_text:
        st.text_area("Derived Rules", st.session_state.rules_text, height=350, disabled=True)

with right:

    st.markdown("### 💬 Step 3 — Ask Questions")

    chat_container = st.container(height=650)

    with chat_container:

        for msg in st.session_state.chat_history:

            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    query = st.chat_input(
        "Ask anything about the document..."
    )

    if query:

        if not st.session_state.doc:

            st.warning(
                "Upload and ingest a document first."
            )

        else:

            # Show user message immediately
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(query)

            with st.spinner("Thinking..."):

                q_type, answer = route_query(
                    query,
                    st.session_state.doc
                )

            labels = {
                "summary": "📝 Document Summary",
                "rule_query": "🔎 Rule Query",
                "factual_qa": "📄 Factual QA",
                "compare": "⚖️ Document Comparison"
            }

            full = (
                f"**[{labels.get(q_type, q_type)}]**\n\n"
                f"{answer}"
            )

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": query
                }
            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": full
                }
            )

            st.rerun()

    if st.session_state.chat_history:

        if st.button(
            "🧹 Clear Memory",
            use_container_width=True
        ):

            st.session_state.chat_history = []

            if st.session_state.doc:
                st.session_state.doc["memory"] = []

            st.rerun()