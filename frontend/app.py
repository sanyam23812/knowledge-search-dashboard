"""
Streamlit dashboard — Search, KPIs, Evaluation, Debug pages.
"""

import sys
import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Make sure backend is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.db import get_all_logs, get_metrics
from app.search.bm25 import BM25Index
from app.search.vector import VectorIndex
from app.search.hybrid import HybridSearcher

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Knowledge Search Dashboard",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load searcher (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_searcher():
    try:
        bm25 = BM25Index()
        bm25.load()
        vec = VectorIndex()
        vec.load()
        return HybridSearcher(bm25, vec)
    except Exception as e:
        return None


searcher = load_searcher()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

page = st.sidebar.radio(
    "Navigate",
    ["🔍 Search", "📊 KPI Dashboard", "🧪 Evaluation", "🐛 Debug Logs"]
)

# ===========================================================================
# PAGE 1 — Search
# ===========================================================================

if page == "🔍 Search":
    st.title("🔍 Knowledge Search")

    if searcher is None:
        st.error("Search indexes not loaded. Run the ingest and index pipeline first.")
        st.code("python -m app.ingest --download\npython -m app.index")
        st.stop()

    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        query = st.text_input("Enter your search query", placeholder="e.g. whale hunting ocean")
    with col2:
        alpha = st.slider("Alpha (BM25 ↔ Vector)", 0.0, 1.0, 0.5, 0.1)
    with col3:
        top_k = st.number_input("Top K", min_value=1, max_value=50, value=10)

    if query:
        with st.spinner("Searching..."):
            results = searcher.search(query, top_k=int(top_k), alpha=alpha)

        st.markdown(f"**{len(results)} results** for `{query}`")
        st.divider()

        for i, r in enumerate(results):
            with st.expander(f"#{i+1} — {r['title']}  |  score: {r['hybrid_score']:.4f}"):
                st.markdown(f"**Highlight:** ...{r['highlight']}...")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Hybrid Score", f"{r['hybrid_score']:.4f}")
                col_b.metric("BM25 Score",   f"{r['norm_bm25']:.4f}")
                col_c.metric("Vector Score", f"{r['norm_vector']:.4f}")
                st.caption(f"doc_id: {r['doc_id']}  |  source: {r.get('source','')}")


# ===========================================================================
# PAGE 2 — KPI Dashboard
# ===========================================================================

elif page == "📊 KPI Dashboard":
    st.title("📊 KPI Dashboard")

    logs = get_all_logs()

    if not logs:
        st.info("No search logs yet. Run some searches first!")
        st.stop()

    df = pd.DataFrame(logs)
    df["created_at"] = pd.to_datetime(df["created_at"])

    # Top metrics row
    metrics = get_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Requests",  metrics["total_requests"])
    c2.metric("p50 Latency (ms)", f"{metrics['p50_ms']:.1f}")
    c3.metric("p95 Latency (ms)", f"{metrics['p95_ms']:.1f}")
    c4.metric("Zero-Result Queries", metrics["zero_results"])

    st.divider()

    # Request volume over time
    st.subheader("Request Volume Over Time")
    vol = df.set_index("created_at").resample("1min").size().reset_index()
    vol.columns = ["time", "count"]
    fig = px.line(vol, x="time", y="count", title="Requests per Minute")
    st.plotly_chart(fig, use_container_width=True)

    # Latency distribution
    st.subheader("Latency Distribution")
    fig2 = px.histogram(df, x="latency_ms", nbins=30, title="Latency (ms)")
    st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)

    # Top queries
    with col1:
        st.subheader("Top Queries")
        top_q = df["query"].value_counts().head(10).reset_index()
        top_q.columns = ["query", "count"]
        st.dataframe(top_q, use_container_width=True)

    # Zero result queries
    with col2:
        st.subheader("Zero-Result Queries")
        zero = df[df["result_count"] == 0][["query", "created_at"]]
        if zero.empty:
            st.success("No zero-result queries!")
        else:
            st.dataframe(zero, use_container_width=True)


# ===========================================================================
# PAGE 3 — Evaluation
# ===========================================================================

elif page == "🧪 Evaluation":
    st.title("🧪 Evaluation Results")

    csv_path = Path("data/metrics/experiments.csv")

    if not csv_path.exists():
        st.info("No experiments run yet.")
        st.code("python -m app.eval --alpha 0.5")
        st.stop()

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Summary metrics
    latest = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest nDCG@10",   f"{latest['ndcg_at_10']:.4f}")
    c2.metric("Latest Recall@10", f"{latest['recall_at_10']:.4f}")
    c3.metric("Latest MRR@10",    f"{latest['mrr_at_10']:.4f}")

    st.divider()

    # nDCG trend
    st.subheader("nDCG@10 Trend Across Experiments")
    fig = px.line(df, x="timestamp", y="ndcg_at_10",
                  markers=True, title="nDCG@10 over runs")
    st.plotly_chart(fig, use_container_width=True)

    # Full experiment table
    st.subheader("All Experiments")
    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)


# ===========================================================================
# PAGE 4 — Debug Logs
# ===========================================================================

elif page == "🐛 Debug Logs":
    st.title("🐛 Debug Logs")

    logs = get_all_logs()

    if not logs:
        st.info("No logs yet.")
        st.stop()

    df = pd.DataFrame(logs)
    df["created_at"] = pd.to_datetime(df["created_at"])

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        show_errors_only = st.checkbox("Show errors only")
    with col2:
        time_range = st.selectbox("Time range", ["All", "Last 1 hour", "Last 24 hours"])

    if show_errors_only:
        df = df[df["error"].notna()]

    if time_range == "Last 1 hour":
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=1)
        df = df[df["created_at"].dt.tz_localize(None) > cutoff]
    elif time_range == "Last 24 hours":
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=24)
        df = df[df["created_at"].dt.tz_localize(None) > cutoff]

    st.dataframe(
        df[["created_at", "query", "latency_ms", "result_count", "alpha", "error"]],
        use_container_width=True
    )