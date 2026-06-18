"""
Agentic RAG — Streamlit
Enhanced Architecture per Enhancement Roadmap

Agents:
  Ingestion Agent     — parse, chunk, embed, metadata extraction, section detection
  Planner Agent       — intent classification (5 types)
  Query Expansion     — multi-query retrieval for better recall
  Reranker            — cross-encoder reranking after vector search
  Rule Extraction     — policies, constraints, compliance (cached)
  Summary Agent       — executive summary, section summary, multi-doc (cached)
  Factual QA Agent    — grounded answers with citations, confidence scoring
  Analysis Agent      — reasoning, synthesis, strengths/risks/findings
  Compare Agent       — cross-document similarities, differences, conflicts
  Metadata Agent      — topics, entities, keywords, sections
  Response Composer   — unified answer + sources
"""

import os, re, json
import streamlit as st
import numpy as np
import faiss

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURATION                                    ║
# ║  Only edit this block to swap providers / models                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

BASE_URL = "https://api.openai.com/v1"
MODEL    = "gpt-4o-mini"
API_KEY  = os.environ.get("OPENAI_API_KEY", "")

# Groq:     BASE_URL = "https://api.groq.com/openai/v1"  | MODEL = "llama3-8b-8192"
# Together: BASE_URL = "https://api.together.xyz/v1"      | MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# Ollama:   BASE_URL = "http://localhost:11434/v1"         | MODEL = "mistral"
# vLLM:     BASE_URL = "http://localhost:8000/v1"          | MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Dynamic top_k per intent
TOP_K = {
    "factual_lookup": 5,
    "summary":        15,
    "compare":        20,
    "document_analysis": 15,
    "rule_query":     10,
}

# ══════════════════════════════════════════════════════════════════════════════
# CORE: LLM + EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_client():
    from openai import OpenAI
    key = st.secrets.get("OPENAI_API_KEY", API_KEY)
    return OpenAI(api_key=key, base_url=BASE_URL)

def llm(system, user, temperature=0.2):
    resp = get_client().chat.completions.create(
        model=MODEL, temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}]
    )
    return resp.choices[0].message.content.strip()

@st.cache_resource
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    vecs = get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype="float32")

@st.cache_resource
def get_reranker():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_pdf(file):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(file)
        pages = []
        for i, p in enumerate(reader.pages):
            text = p.extract_text() or ""
            if text.strip():
                pages.append({"page": i+1, "text": text})
        return pages
    except Exception as e:
        return [{"page": 1, "text": f"[PDF error: {e}]"}]

def parse_docx(file):
    try:
        from docx import Document
        doc = Document(file)
        return [{"page": 1, "text": "\n".join(p.text for p in doc.paragraphs)}]
    except Exception as e:
        return [{"page": 1, "text": f"[DOCX error: {e}]"}]

def parse_document(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    if ext == ".pdf":  return parse_pdf(uploaded_file)
    if ext == ".docx": return parse_docx(uploaded_file)
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    return [{"page": 1, "text": text}]

# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(pages, source, chunk_size=400, overlap=80):
    chunks = []
    chunk_id = 0
    for page_data in pages:
        words = page_data["text"].split()
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append({
                "source":   source,
                "chunk_id": chunk_id,
                "page":     page_data["page"],
                "text":     chunk
            })
            chunk_id += 1
            i += chunk_size - overlap
    return chunks

# ══════════════════════════════════════════════════════════════════════════════
# FAISS + RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def build_index(all_chunks):
    texts = [c["text"] for c in all_chunks]
    embeddings = embed(texts)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query, index, chunks, top_k=10):
    scores, idxs = index.search(embed([query]), min(top_k, len(chunks)))
    return [
        {**chunks[idx], "score": float(score)}
        for score, idx in zip(scores[0], idxs[0])
        if idx >= 0
    ]

def retrieve_multi_query(queries, index, chunks, top_k=10):
    """Query expansion: retrieve for multiple queries, deduplicate by chunk_id+source."""
    seen, results = set(), []
    for q in queries:
        for r in retrieve(q, index, chunks, top_k=top_k):
            key = (r["source"], r["chunk_id"])
            if key not in seen:
                seen.add(key)
                results.append(r)
    return results

def rerank(query, results, top_k=10):
    """Cross-encoder reranking. Falls back gracefully if model unavailable."""
    reranker = get_reranker()
    if not reranker or len(results) <= top_k:
        return results[:top_k]
    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    for score, r in ranked:
        r["rerank_score"] = float(score)
    return [r for _, r in ranked[:top_k]]

def compute_confidence(results):
    """Return a confidence signal based on top score and source diversity."""
    if not results:
        return 0.0
    top_score = results[0]["score"]
    sources = len(set(r["source"] for r in results))
    diversity_bonus = min(sources * 0.05, 0.2)
    return min(top_score + diversity_bonus, 1.0)

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — INGESTION + METADATA
# ══════════════════════════════════════════════════════════════════════════════

METADATA_SYSTEM = """You are a Metadata Extraction Agent.
Analyze the document content and return a JSON object with exactly these keys:
{
  "document_type": "resume | research_paper | contract | report | policy | manual | invoice | other",
  "topics": ["topic1", "topic2"],
  "entities": ["entity1", "entity2"],
  "keywords": ["kw1", "kw2"],
  "sections": ["section1", "section2"]
}
Return ONLY the JSON object. No preamble, no markdown fences."""

def metadata_agent(raw_text):
    sample = " ".join(raw_text.split()[:800])
    raw = llm(METADATA_SYSTEM, f"Document content:\n{sample}")
    try:
        return json.loads(re.sub(r"```(?:json)?|```", "", raw).strip())
    except Exception:
        return {"document_type": "unknown", "topics": [], "entities": [], "keywords": [], "sections": []}

def ingestion_agent(uploaded_files):
    all_chunks = []
    documents  = {}
    metadata   = {}

    for file in uploaded_files:
        pages  = parse_document(file)
        chunks = chunk_text(pages, file.name)
        documents[file.name] = chunks
        all_chunks.extend(chunks)

        # Metadata extraction per document
        full_text = " ".join(p["text"] for p in pages)
        metadata[file.name] = metadata_agent(full_text)

    index = build_index(all_chunks)

    return {
        "chunks":    all_chunks,
        "documents": documents,
        "index":     index,
        "metadata":  metadata,   # per-doc metadata
        "summary":   {},         # cached summaries
        "rules":     {},         # cached rules
        "memory":    []
    }, f"Loaded {len(uploaded_files)} document(s), {len(all_chunks)} chunks total."

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — PLANNER (5-way intent classification)
# ══════════════════════════════════════════════════════════════════════════════

PLANNER_SYSTEM = """You are a Query Planner Agent.
Classify the user query into exactly one intent:

- "factual_lookup"     — specific facts, names, dates, values, definitions
- "summary"            — summarize, overview, gist, what is this about
- "compare"            — compare, difference, versus, vs, similarities
- "document_analysis"  — what stands out, strengths, risks, findings, most important
- "rule_query"         — rules, policies, constraints, allowed, not allowed, must, shall

Reply with ONLY the intent string."""

def planner_agent(query):
    q = query.lower()
    # Fast keyword pre-checks before LLM call
    if any(k in q for k in ["summarize", "summary", "overview", "gist", "what is this about", "brief", "takeaway"]):
        return "summary"
    if any(k in q for k in ["compare", "difference", "differences", "versus", " vs ", "similarities", "contrast"]):
        return "compare"
    if any(k in q for k in ["stands out", "strength", "weakness", "risk", "finding", "most important", "key insight", "analysis"]):
        return "document_analysis"
    if any(k in q for k in ["rule", "policy", "policies", "constraint", "allowed", "not allowed", "must", "shall", "prohibited", "compliance"]):
        return "rule_query"
    result = llm(PLANNER_SYSTEM, query, temperature=0).strip().lower()
    return result if result in ["factual_lookup", "summary", "compare", "document_analysis", "rule_query"] else "factual_lookup"

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — QUERY EXPANSION
# ══════════════════════════════════════════════════════════════════════════════

EXPANSION_SYSTEM = """You are a Query Expansion Agent.
Given a user query, generate 4-6 semantically similar retrieval queries that would help find relevant content.
Return ONLY a JSON array of strings (the expanded queries).
No preamble, no markdown fences."""

def query_expansion_agent(query):
    raw = llm(EXPANSION_SYSTEM, f"Query: {query}", temperature=0.3)
    try:
        expanded = json.loads(re.sub(r"```(?:json)?|```", "", raw).strip())
        if isinstance(expanded, list):
            return [query] + [str(q) for q in expanded[:5]]
    except Exception:
        pass
    return [query]

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — RULE EXTRACTION (cached per doc)
# ══════════════════════════════════════════════════════════════════════════════

RULE_SYSTEM = """You are a Rule Extraction Agent.
Extract ALL explicit and implicit rules, constraints, policies, compliance requirements, and obligations.
Return ONLY a valid JSON array of strings. Each string = one self-contained rule in plain English.
No preamble, no markdown fences — just the raw JSON array."""

def rule_extraction_agent(doc, source=None):
    # Return cached if available
    cache_key = source or "__all__"
    if cache_key in doc.get("rules", {}):
        return doc["rules"][cache_key]

    chunks = doc["chunks"]
    if source:
        chunks = [c for c in chunks if c["source"] == source]

    queries = ["rules regulations policy must shall", "not allowed forbidden prohibited", "requirements constraints compliance obligations"]
    candidates = retrieve_multi_query(queries, doc["index"],
                                      doc["chunks"] if not source else chunks,
                                      top_k=8)
    top_chunks = [r["text"] for r in candidates[:15]]
    raw = llm(RULE_SYSTEM, "Document content:\n" + "\n\n---\n\n".join(top_chunks))
    try:
        rules = json.loads(re.sub(r"```(?:json)?|```", "", raw).strip())
        if isinstance(rules, list):
            result = [str(r) for r in rules]
        else:
            raise ValueError
    except Exception:
        result = [l.strip("•- ").strip() for l in raw.splitlines() if l.strip()]

    doc["rules"][cache_key] = result
    return result

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 5 — SUMMARY AGENT (cached per doc)
# ══════════════════════════════════════════════════════════════════════════════

SUMMARY_SYSTEM = """You are a Document Summarization Agent.
Generate a structured summary with:
1. **Executive Summary** (2-3 sentences)
2. **Key Topics**
3. **Important Findings / Key Points**
4. **Notable Entities / Numbers / Dates**

Be concise and factual. Only use information present in the provided content."""

def summary_agent(doc, source=None):
    cache_key = source or "__all__"
    if cache_key in doc.get("summary", {}):
        return doc["summary"][cache_key]

    if source and source in doc["documents"]:
        chunks = [c["text"] for c in doc["documents"][source][:20]]
    else:
        chunks = [c["text"] for c in doc["chunks"][:20]]

    result = llm(SUMMARY_SYSTEM, "\n\n".join(chunks))
    doc["summary"][cache_key] = result
    return result

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 6 — FACTUAL QA AGENT
# ══════════════════════════════════════════════════════════════════════════════

FACTUAL_QA_SYSTEM = """You are a precise Factual QA Agent. Answer strictly from the provided document context.

Rules:
1. Answer ONLY from the given chunks. Do not infer beyond what is stated.
2. End every answer with a "📎 Sources" section listing: source filename, page number, chunk ID.
3. If the answer is not found, say "Not found in document."
4. Be concise and direct.

Format:
<your answer>

📎 Sources: [filename | Page X | Chunk #Y], ..."""

def factual_qa_agent(query, doc):
    queries   = query_expansion_agent(query)
    top_k     = TOP_K["factual_lookup"]
    candidates = retrieve_multi_query(queries, doc["index"], doc["chunks"], top_k=top_k*3)
    results   = rerank(query, candidates, top_k=top_k)

    confidence = compute_confidence(results)
    if confidence < 0.05:
        return "⚠️ Low confidence — the document may not contain relevant information for this query. Try rephrasing or specifying the document."

    context = "\n\n".join(
        f"[{r['source']} | Page {r.get('page',1)} | Chunk #{r['chunk_id']} | score={r['score']:.2f}]\n{r['text']}"
        for r in results
    )
    rules = doc.get("rules", {}).get("__all__", [])
    rules_text = ("\n\nDerived Rules:\n" + "\n".join(
        f"[Rule {i+1}] {r}" for i, r in enumerate(rules)
    )) if rules else ""

    memory = doc.get("memory", [])
    history = ""
    if memory:
        history = "\nConversation History:\n" + "".join(
            f"User: {t['q']}\nAssistant: {t['a']}\n\n" for t in memory[-4:]
        )

    answer = llm(FACTUAL_QA_SYSTEM, f"{history}Context:\n{context}{rules_text}\n\nQuestion: {query}")
    doc["memory"].append({"q": query, "a": answer})
    return answer

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 7 — ANALYSIS AGENT
# ══════════════════════════════════════════════════════════════════════════════

ANALYSIS_SYSTEM = """You are a Document Analysis Agent. You are allowed to reason, synthesize, and infer.

Your task:
1. Analyze the provided document content deeply.
2. Answer the user's analytical question with structured reasoning.
3. Use headings where helpful (Strengths, Risks, Key Findings, etc.)
4. End with a "📎 Sources" section listing chunks used.

Do not make up facts — ground your analysis in the provided context."""

def analysis_agent(query, doc):
    queries    = query_expansion_agent(query)
    top_k      = TOP_K["document_analysis"]
    candidates = retrieve_multi_query(queries, doc["index"], doc["chunks"], top_k=top_k*2)
    results    = rerank(query, candidates, top_k=top_k)

    context = "\n\n".join(
        f"[{r['source']} | Page {r.get('page',1)} | Chunk #{r['chunk_id']}]\n{r['text']}"
        for r in results
    )
    memory = doc.get("memory", [])
    history = ""
    if memory:
        history = "\nConversation History:\n" + "".join(
            f"User: {t['q']}\nAssistant: {t['a']}\n\n" for t in memory[-4:]
        )
    answer = llm(ANALYSIS_SYSTEM, f"{history}Context:\n{context}\n\nQuestion: {query}")
    doc["memory"].append({"q": query, "a": answer})
    return answer

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 8 — COMPARE AGENT
# ══════════════════════════════════════════════════════════════════════════════

COMPARE_SYSTEM = """You are a Document Comparison Agent.

Compare information across the provided document chunks.
Structure your response as:
1. **Similarities**
2. **Differences**
3. **Conflicts / Contradictions**
4. **Overlaps**
5. **Summary**

Cite which document each point comes from. Be factual."""

def compare_agent(query, doc):
    queries    = query_expansion_agent(query)
    top_k      = TOP_K["compare"]
    candidates = retrieve_multi_query(queries, doc["index"], doc["chunks"], top_k=top_k*2)
    results    = rerank(query, candidates, top_k=top_k)

    context = "\n\n".join(
        f"[{r['source']} | Chunk #{r['chunk_id']}]\n{r['text']}"
        for r in results
    )
    answer = llm(COMPARE_SYSTEM, context + f"\n\nComparison Request: {query}")
    doc["memory"].append({"q": query, "a": answer})
    return answer

# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE COMPOSER — unified routing
# ══════════════════════════════════════════════════════════════════════════════

def normalize(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())

def find_document(query, doc):
    q = normalize(query)
    for filename in doc["documents"]:
        clean = normalize(filename.rsplit(".", 1)[0])
        if clean in q:
            return filename
    return None

def route_and_respond(query, doc):
    """
    Planner → (Query Expansion) → Retrieval → Rerank → Specialized Agent → Response
    Returns (intent_label, answer)
    """
    intent = planner_agent(query)

    # ── Summary ───────────────────────────────────────────────────────────────
    if intent == "summary":
        target = find_document(query, doc)
        if target is None and len(doc["documents"]) == 1:
            target = next(iter(doc["documents"]))
        if target is None:
            docs_list = "\n".join(f"• {d}" for d in doc["documents"])
            return "summary", f"Multiple documents found. Please specify which one:\n\n{docs_list}"
        return "summary", summary_agent(doc, source=target)

    # ── Compare ───────────────────────────────────────────────────────────────
    if intent == "compare":
        return "compare", compare_agent(query, doc)

    # ── Document Analysis ─────────────────────────────────────────────────────
    if intent == "document_analysis":
        return "document_analysis", analysis_agent(query, doc)

    # ── Rule Query ────────────────────────────────────────────────────────────
    if intent == "rule_query":
        if "__all__" not in doc.get("rules", {}):
            doc["rules"]["__all__"] = rule_extraction_agent(doc)
        return "rule_query", factual_qa_agent(query, doc)

    # ── Factual Lookup (default) ──────────────────────────────────────────────
    return "factual_lookup", factual_qa_agent(query, doc)

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")
st.title("🤖 DOCUMENT CHAT")
st.caption("Planner → Query Expansion → Retrieval → Reranking → Specialized Agents → Response Composer")

# Session state
for key, default in [("doc", None), ("rules_text", ""), ("chat_history", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 2])

with left:
    st.markdown("### 📁 Step 1 — Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF / DOCX / TXT (multiple allowed)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button("⚡ Ingest Documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one file.")
        else:
            with st.spinner("Parsing, embedding, extracting metadata..."):
                doc, msg = ingestion_agent(uploaded_files)
            if doc:
                st.session_state.doc = doc
                st.session_state.rules_text = ""

                # Build welcome message with metadata per doc
                meta_lines = []
                for fname, meta in doc["metadata"].items():
                    doc_type = meta.get("document_type", "unknown")
                    topics   = ", ".join(meta.get("topics", [])[:4]) or "—"
                    sections = ", ".join(meta.get("sections", [])[:4]) or "—"
                    meta_lines.append(
                        f"**{fname}**\n"
                        f"  • Type: `{doc_type}`\n"
                        f"  • Topics: {topics}\n"
                        f"  • Sections: {sections}"
                    )

                st.session_state.chat_history = [{
                    "role": "assistant",
                    "content": (
                        f"✅ Loaded **{len(doc['documents'])}** document(s), "
                        f"**{len(doc['chunks'])}** chunks.\n\n"
                        + "\n\n".join(meta_lines)
                        + "\n\nYou can now ask questions, request summaries, compare documents, or extract rules."
                    )
                }]
                st.success(msg)
            else:
                st.error(msg)

    st.divider()

    # Step 2 — Rules
    st.markdown("### 📋 Step 2 — Extract Rules")
    if st.button("🔍 Run Rule Extraction Agent", use_container_width=True):
        if not st.session_state.doc:
            st.warning("Ingest a document first.")
        else:
            with st.spinner("Extracting rules..."):
                rules = rule_extraction_agent(st.session_state.doc)
            st.session_state.doc["rules"]["__all__"] = rules
            st.session_state.rules_text = "\n".join(
                f"{i+1}. {r}" for i, r in enumerate(rules)
            ) if rules else "No rules found."
            st.success(f"Extracted **{len(rules)} rules**.")

    if st.session_state.rules_text:
        st.text_area("Derived Rules", st.session_state.rules_text, height=320, disabled=True)

    # Metadata panel
    if st.session_state.doc:
        st.divider()
        st.markdown("### 🗂️ Document Metadata")
        for fname, meta in st.session_state.doc["metadata"].items():
            with st.expander(fname, expanded=False):
                st.markdown(f"**Type:** `{meta.get('document_type','—')}`")
                st.markdown(f"**Topics:** {', '.join(meta.get('topics',[]))  or '—'}")
                st.markdown(f"**Entities:** {', '.join(meta.get('entities',[])[:8]) or '—'}")
                st.markdown(f"**Keywords:** {', '.join(meta.get('keywords',[])[:8]) or '—'}")
                st.markdown(f"**Sections:** {', '.join(meta.get('sections',[]))  or '—'}")

with right:
    st.markdown("### 💬 Step 3 — Ask Questions")

    chat_container = st.container(height=650)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    query = st.chat_input("Ask anything — summary, compare, analysis, rules, factual questions...")

    if query:
        if not st.session_state.doc:
            st.warning("Upload and ingest a document first.")
        else:
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(query)

            with st.spinner("Planning → Retrieving → Reranking → Answering..."):
                intent, answer = route_and_respond(query, st.session_state.doc)

            labels = {
                "summary":           "📝 Summary",
                "rule_query":        "🔎 Rule Query",
                "factual_lookup":    "📄 Factual QA",
                "compare":           "⚖️ Comparison",
                "document_analysis": "🔬 Analysis",
            }
            full = f"**[{labels.get(intent, intent)}]**\n\n{answer}"

            st.session_state.chat_history.append({"role": "user",      "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": full})
            st.rerun()

    if st.session_state.chat_history:
        if st.button("🧹 Clear Memory", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.doc:
                st.session_state.doc["memory"] = []
            st.rerun()
