# Decision Log

Key design decisions made during development.

---

## 1. Normalisation Strategy — Min-Max (default)

**Options considered:** min-max normalisation vs z-score normalisation

**Decision:** Use min-max as default, with z-score available via flag.

**Reasoning:**
Min-max maps all scores to [0,1] which makes the alpha blending
formula intuitive — alpha=0.5 gives exactly equal weight to both
signals. Z-score can produce negative values before shifting, which
is less intuitive for end users inspecting score breakdowns.
Both are implemented and selectable via the norm parameter.

---

## 2. Embedding Model — all-MiniLM-L6-v2

**Options considered:** all-MiniLM-L6-v2, all-mpnet-base-v2,
paraphrase-MiniLM-L3-v2

**Decision:** all-MiniLM-L6-v2

**Reasoning:**
Best balance of speed and quality for CPU-only environments.
Produces 384-dimensional embeddings and runs inference in under
1 second per query on a typical laptop. all-mpnet-base-v2 is more
accurate but 3x slower on CPU. paraphrase-MiniLM-L3-v2 is faster
but noticeably weaker on retrieval benchmarks.

---

## 3. Vector Search — FAISS IndexFlatIP

**Options considered:** FAISS IndexFlatIP, FAISS IndexIVFFlat, hnswlib

**Decision:** FAISS IndexFlatIP with L2 normalisation (cosine similarity)

**Reasoning:**
For a corpus of ~300-500 chunked documents, an exact search index
is fast enough on CPU (under 10ms per query). Approximate indexes
like IVFFlat or HNSW add complexity and only help at 100k+ docs.
L2-normalised inner product is equivalent to cosine similarity,
which is the standard for sentence-transformer embeddings.

---

## 4. Frontend — Streamlit over React+Vite

**Options considered:** React + Vite, Streamlit

**Decision:** Streamlit

**Reasoning:**
Streamlit allows rapid dashboard development in pure Python,
avoiding the need to maintain a separate Node.js frontend and
a REST-to-frontend data layer. Since the backend is already Python,
Streamlit can import modules directly, reducing complexity.
The tradeoff is less UI flexibility, which is acceptable for an
internal KPI dashboard.

---

## 5. Storage — SQLite over PostgreSQL

**Options considered:** SQLite, PostgreSQL, MongoDB

**Decision:** SQLite

**Reasoning:**
The assignment requires CPU-only local execution with no paid
cloud services. SQLite requires zero setup, is file-based, and
handles the expected query log volume (thousands of rows) with
no performance issues. PostgreSQL would add unnecessary ops overhead
for a single-developer local system.