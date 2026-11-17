"""Streamlit front end.   streamlit run app.py

Evidence-Grounded Research Document Citation Engine.
Upload PDFs in the sidebar, then use one of the three modes. Every statement shows its label,
its numbered evidence, and the exact passage behind it; removed claims are audited in a verification log.
"""
from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from egc.config import Settings
from egc.pipeline import Engine
from egc.render import NO_ANSWER, ieee_reference, numbering
from egc.schema import Answer

st.set_page_config(
    page_title="Evidence-Grounded Research Engine",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Professional SaaS / Research Lab Design System
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-main: #0b0f17;
    --bg-surface: #111827;
    --bg-surface-elevated: #1a2234;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-focus: #3b82f6;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --color-primary: #2563eb;
    --color-primary-hover: #1d4ed8;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

.stApp {
    background-color: #0b0f17 !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(37, 99, 235, 0.07) 0px, transparent 45%),
        radial-gradient(at 100% 100%, rgba(14, 165, 233, 0.05) 0px, transparent 45%);
    background-attachment: fixed !important;
    color: #f8fafc !important;
}

/* Clean Header Navbar */
.egc-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    margin-bottom: 22px;
}
.egc-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}
.egc-logo {
    font-size: 24px;
    background: rgba(37, 99, 235, 0.15);
    border: 1px solid rgba(37, 99, 235, 0.35);
    padding: 6px 10px;
    border-radius: 8px;
    line-height: 1;
}
.egc-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.egc-subtitle {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 3px;
}
.egc-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #94a3b8;
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 6px rgba(16, 185, 129, 0.6);
}

/* Metric Cards */
.metric-card {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 14px 18px;
    transition: all 0.15s ease;
}
.metric-card:hover {
    border-color: rgba(255, 255, 255, 0.16);
    background: #141d2e;
}
.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #64748b;
}
.metric-val {
    font-size: 1.55rem;
    font-weight: 700;
    color: #ffffff;
    margin-top: 4px;
    line-height: 1;
}
.metric-sub {
    font-size: 0.8rem;
    font-weight: 400;
    color: #94a3b8;
}
.text-success { color: #34d399 !important; }
.text-danger { color: #f87171 !important; }
.text-info { color: #38bdf8 !important; }
.text-primary { color: #60a5fa !important; }

/* Statement Cards */
.statement-card {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    transition: background 0.15s ease;
}
.statement-card:hover {
    background: #141c2c;
    border-color: rgba(255, 255, 255, 0.14);
}
.statement-card.card-finding {
    border-left-color: #10b981;
}
.statement-card.card-limitation {
    border-left-color: #ef4444;
}
.statement-card.card-interpretation {
    border-left-color: #f59e0b;
}
.statement-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.statement-body {
    font-size: 0.96rem;
    line-height: 1.6;
    color: #e2e8f0;
}
.statement-legend {
    font-size: 0.82rem;
    color: #64748b;
    padding: 10px 14px;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    margin: 14px 0 20px 0;
}

/* Badges */
.egc-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.badge-finding {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.badge-limitation {
    background: rgba(239, 68, 68, 0.12);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.badge-interpretation {
    background: rgba(245, 158, 11, 0.12);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* Citation Pill */
.cite-pill {
    display: inline-block;
    padding: 1px 6px;
    margin-left: 4px;
    background: rgba(59, 130, 246, 0.14);
    color: #93c5fd;
    border: 1px solid rgba(59, 130, 246, 0.32);
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* Evidence Box */
.evidence-box {
    padding: 12px 16px;
    background: rgba(0, 0, 0, 0.28);
    border-radius: 6px;
    border-left: 2px solid #3b82f6;
    margin-top: 8px;
}
.evidence-meta {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-bottom: 6px;
    font-weight: 500;
}
.evidence-text {
    font-size: 0.92rem;
    line-height: 1.6;
    color: #cbd5e1;
}
.evidence-highlight {
    background: rgba(245, 158, 11, 0.22);
    color: #fef08a;
    padding: 2px 4px;
    border-radius: 3px;
    border-bottom: 2px solid #f59e0b;
}

/* Audit / Removed Cards */
.audit-note {
    font-size: 0.86rem;
    color: #94a3b8;
    margin-bottom: 12px;
    line-height: 1.5;
}
.removed-card {
    background: rgba(239, 68, 68, 0.04);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-left: 3px solid #ef4444;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.removed-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
}
.removed-reason {
    font-size: 0.78rem;
    color: #fca5a5;
    font-weight: 500;
}
.removed-text {
    font-size: 0.92rem;
    color: #94a3b8;
    line-height: 1.5;
}

/* Reference items */
.reference-item {
    display: flex;
    gap: 10px;
    padding: 10px 14px;
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    line-height: 1.55;
    color: #cbd5e1;
}
.ref-num {
    font-weight: 700;
    color: #60a5fa;
    font-family: 'JetBrains Mono', monospace;
}

/* Suggested Queries */
.prompt-chips-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #94a3b8;
    margin-bottom: 6px;
    margin-top: 10px;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0c1017 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
.sidebar-header {
    font-size: 0.95rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
}
.sidebar-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin-bottom: 16px;
}

/* Baseline Mode */
.baseline-warning {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.28);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.84rem;
    color: #fde68a;
    margin-bottom: 16px;
    line-height: 1.5;
}
.baseline-stmt {
    font-size: 0.93rem;
    color: #cbd5e1;
    margin-bottom: 10px;
    line-height: 1.6;
}
.cite-pill-plain {
    display: inline-block;
    padding: 1px 5px;
    background: rgba(255, 255, 255, 0.06);
    color: #94a3b8;
    border-radius: 4px;
    font-size: 0.75rem;
    font-family: monospace;
}
.reference-item-plain {
    font-size: 0.84rem;
    color: #94a3b8;
    margin-bottom: 6px;
}

/* Form Inputs & Buttons */
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
    background: #111620 !important;
    border: 1px solid #232d3f !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
    font-size: 0.92rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
}

.stButton button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 0.15s ease !important;
}
.stButton button[kind="primary"], .stButton button[type="primary"] {
    background: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
}
.stButton button[kind="primary"]:hover, .stButton button[type="primary"]:hover {
    background: #1d4ed8 !important;
    border-color: #60a5fa !important;
}
.stButton button:not([kind="primary"]):not([type="primary"]) {
    background: #161d2b !important;
    border: 1px solid #243044 !important;
    color: #cbd5e1 !important;
}
.stButton button:not([kind="primary"]):not([type="primary"]):hover {
    background: #1e293b !important;
    border-color: #384964 !important;
    color: #ffffff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: #111622;
    padding: 4px;
    border-radius: 8px;
    border: 1px solid #202b3d;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    padding: 7px 16px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    color: #94a3b8 !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: #111827 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}
.streamlit-expanderHeader:hover {
    border-color: rgba(255, 255, 255, 0.18) !important;
}
[data-testid="stExpanderDetails"] {
    background: #0d121c !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 14px !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

UPLOADS = Path("papers")
COLOR = {"finding": "green", "limitation": "red", "interpretation": "orange"}
NAME = {"finding": "Finding", "limitation": "Limitation", "interpretation": "Interpretation"}


@st.cache_resource(show_spinner="Loading index and embedding model...")
def get_engine() -> Engine:
    return Engine(Settings())


def render_header(paper_count: int, chunk_count: int) -> None:
    st.markdown(
        f"""
        <div class="egc-navbar">
            <div class="egc-brand">
                <div class="egc-logo">📑</div>
                <div>
                    <div class="egc-title">Evidence-Grounded Research Engine</div>
                    <div class="egc-subtitle">Zero-Hallucination Retrieval, Attribution & Quote-Anchored Verification</div>
                </div>
            </div>
            <div class="egc-status-pill">
                <span class="status-dot"></span>
                <span>Hybrid Index Active • <b>{paper_count}</b> Papers ({chunk_count} Chunks)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_answer(ans: Answer) -> None:
    if ans.insufficient and not ans.statements:
        st.warning(NO_ANSWER)
        return

    ev, ref = numbering(ans)
    by_tag = {p.tag: p for p in ans.passages}

    # Metric Banner
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Verified Statements</div>
            <div class="metric-val text-success">{len(ans.statements)} <span class="metric-sub">kept</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Audit Filtered</div>
            <div class="metric-val text-danger">{len(ans.removed)} <span class="metric-sub">purged</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pipeline LLM Calls</div>
            <div class="metric-val text-info">{ans.llm_calls} <span class="metric-sub">calls</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Execution Latency</div>
            <div class="metric-val text-primary">{ans.timings.get('total', 0):.2f}s</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Statements Section
    st.markdown("#### 📋 Verified Statements")
    heading = None
    for st_ in ans.statements:
        if st_.heading and st_.heading != heading:
            heading = st_.heading
            st.markdown(f"##### {heading}")

        badge_class = f"badge-{st_.label}"
        badge_name = NAME.get(st_.label, st_.label.title())
        cite_html = "".join([f'<span class="cite-pill">[{ev[t]}]</span>' for t in st_.cites if t in ev])

        st.markdown(f"""
        <div class="statement-card card-{st_.label}">
            <div class="statement-header">
                <span class="egc-badge {badge_class}">{badge_name}</span>
            </div>
            <div class="statement-body">{html.escape(st_.text)} {cite_html}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="statement-legend">
            <b>Taxonomy:</b> &nbsp;
            <span class="egc-badge badge-finding">Finding</span> = Stated by the authors in the paper &nbsp;•&nbsp; 
            <span class="egc-badge badge-limitation">Limitation</span> = Limitation explicitly noted by the paper &nbsp;•&nbsp; 
            <span class="egc-badge badge-interpretation">Interpretation</span> = System synthesis / inference
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Evidence Drawer / Inspector
    st.markdown("#### 🔍 Cited Evidence & Passages")
    quotes = {s.tag: s.quote for x in ans.statements for s in x.support if s.quote}
    for tag, n in ev.items():
        p = by_tag[tag]
        with st.expander(f"[{n}] {p.paper.short()} · Pages: {p.chunk.pages} · Section: {p.chunk.section or 'n/a'}"):
            text = html.escape(p.chunk.text)
            quote = html.escape(quotes.get(tag, ""))
            if quote and quote in text:
                text = text.replace(quote, f'<mark class="evidence-highlight">{quote}</mark>')
            st.markdown(f"""
            <div class="evidence-box">
                <div class="evidence-meta">Source: <b>{html.escape(p.paper.title)}</b> · Chunk ID: {p.chunk.chunk_id}</div>
                <div class="evidence-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    # Removed Statements (Audit Trail)
    if ans.removed:
        st.markdown("#### 🛡️ Verification Audit Trail (Rejected Claims)")
        st.markdown("""
        <div class="audit-note">
            The following candidate statements were generated during initial drafting but <b>failed rigorous quote-anchored verification</b> against the source passages. They were purged from the answer to prevent hallucination.
        </div>
        """, unsafe_allow_html=True)
        for s in ans.removed:
            badge_class = f"badge-{s.label}"
            badge_name = NAME.get(s.label, s.label.title())
            st.markdown(f"""
            <div class="removed-card">
                <div class="removed-header">
                    <span class="egc-badge {badge_class}">{badge_name}</span>
                    <span class="removed-reason">❌ Rejection Reason: {html.escape(s.reason)}</span>
                </div>
                <div class="removed-text"><s>{html.escape(s.text)}</s></div>
            </div>
            """, unsafe_allow_html=True)

    # Bibliographic References
    st.markdown("#### 📚 Bibliographic References (IEEE Format)")
    papers = {p.paper.paper_id: p.paper for p in ans.passages}
    for pid, n in ref.items():
        st.markdown(f"""
        <div class="reference-item">
            <span class="ref-num">[{n}]</span>
            <span class="ref-text">{ieee_reference(papers[pid])}</span>
        </div>
        """, unsafe_allow_html=True)


def show_baseline(out: dict) -> None:
    st.markdown("""
    <div class="baseline-warning">
        ⚠️ <b>Standard Unchecked LLM Output</b><br>
        Generated from parametric memory without document retrieval or quote-anchored verification. References below may contain fabricated or inaccurate metadata.
    </div>
    """, unsafe_allow_html=True)
    for s in out["statements"]:
        refs = " ".join([f'<span class="cite-pill-plain">[{n}]</span>' for n in s["refs"]])
        st.markdown(f'<div class="baseline-stmt">• {html.escape(s["text"])} {refs}</div>', unsafe_allow_html=True)
    st.markdown("##### Parametric References")
    for r in out["references"]:
        st.markdown(f'<div class="reference-item-plain">{r["n"]}. {html.escape(r["authors"])}, “{html.escape(r["title"])}”, {html.escape(r["venue"])}, {r["year"] or "n.d."}.</div>', unsafe_allow_html=True)


def run(mode: str, compare: bool, **kw) -> None:
    engine = get_engine()
    try:
        with st.spinner("Retrieving, generating, and verifying evidence..."):
            ans = {"ask": engine.ask, "explain": engine.explain, "summarize": engine.summarize}[mode](**kw)
        if compare:
            left, right = st.columns(2)
            with left:
                st.subheader("🛡️ Evidence-Grounded (Verified)")
                show_answer(ans)
            with right:
                st.subheader("⚠️ Plain LLM (Baseline)")
                with st.spinner("Querying ungrounded model..."):
                    show_baseline(engine.baseline(mode, question=kw.get("question", ""), paper_key=kw.get("paper_key", "")))
        else:
            show_answer(ans)
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")


engine = get_engine()
all_papers = engine.papers()
total_chunks = sum(p.n_chunks for p in all_papers)

# ---------------------------------------------------------
# Sidebar: Paper Library & Knowledge Base
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-header">📚 Research Paper Library</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-sub">{len(all_papers)} papers indexed • {total_chunks} total chunks</div>', unsafe_allow_html=True)

    with st.expander("➕ Ingest New Paper (PDF)", expanded=False):
        files = st.file_uploader("Upload PDF Documents", type="pdf", accept_multiple_files=True)
        enrich = st.checkbox("Query online metadata (Crossref / arXiv)", value=False)
        if files and st.button("Start Ingestion", type="primary", use_container_width=True):
            UPLOADS.mkdir(exist_ok=True)
            for f in files:
                path = UPLOADS / f.name
                path.write_bytes(f.getbuffer())
                try:
                    paper = engine.ingest(path, enrich=enrich)
                    st.success(f"Indexed: {paper.title[:50]}... ({paper.n_chunks} chunks)")
                except Exception as exc:
                    st.error(f"{f.name}: {exc}")
            st.rerun()

    # Paper Filter
    filter_q = st.text_input("Filter indexed papers", placeholder="Search title or authors...", label_visibility="collapsed")
    filtered_papers = [p for p in all_papers if filter_q.lower() in p.title.lower() or any(filter_q.lower() in a.lower() for a in p.authors)] if filter_q else all_papers

    st.markdown("---")
    st.caption(f"INDEXED PAPERS ({len(filtered_papers)})")

    for paper in filtered_papers:
        with st.expander(f"{paper.short()}: {paper.title[:55]}..."):
            st.markdown(f"**Title:** {paper.title}")
            st.markdown(f"**Authors:** {', '.join(paper.authors) or 'Not found'}")
            st.markdown(f"**Year:** {paper.year or 'n.d.'} &nbsp;|&nbsp; **DOI:** {paper.doi or 'None'}")
            st.caption(f"{paper.n_pages} pages · {paper.n_chunks} chunks · Source: {paper.meta_source}")
            if st.button("Delete Paper", key=f"del-{paper.paper_id}", use_container_width=True):
                engine.store.remove_paper(paper.paper_id)
                st.rerun()

    st.markdown("---")
    compare = st.checkbox("Compare with Plain LLM (Baseline)", value=False, help="Runs side-by-side comparison against an ungrounded LLM to demonstrate hallucination reduction.")

# ---------------------------------------------------------
# Main Workspace
# ---------------------------------------------------------
render_header(len(all_papers), total_chunks)

if not all_papers:
    st.info("No research papers in the database. Upload papers in the sidebar to begin.")
else:
    titles = {p.paper_id: f"{p.title[:85]}... ({p.short()})" for p in all_papers}
    ask_tab, explain_tab, summary_tab = st.tabs(["🔍 Ask Research", "📖 Explain Paper", "📝 Summarize Paper"])

    with ask_tab:
        if "ask_input" not in st.session_state:
            st.session_state["ask_input"] = ""

        # Suggested Queries Chips
        st.markdown('<div class="prompt-chips-label">💡 Suggested Research Queries:</div>', unsafe_allow_html=True)
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            if st.button("⚡ Hybrid Search vs Dense Search", use_container_width=True):
                st.session_state["ask_input"] = "How does hybrid retrieval improve search accuracy over dense vector or BM25 alone?"
        with col_q2:
            if st.button("🛡️ LLM Hallucinations in Research", use_container_width=True):
                st.session_state["ask_input"] = "What are the primary causes of hallucination and fabricated citations in large language models?"
        with col_q3:
            if st.button("🔍 Quote Anchoring Mechanism", use_container_width=True):
                st.session_state["ask_input"] = "How does quote anchoring verify that a citation actually supports its claim?"

        question = st.text_area(
            "Research Query across All Papers",
            value=st.session_state.get("ask_input", ""),
            height=90,
            placeholder="Type your technical question here (e.g. How does reciprocal rank fusion combine rankings?)...",
        )

        col_btn, _ = st.columns([1, 4])
        with col_btn:
            ask_pressed = st.button("Run Verified Search", type="primary", use_container_width=True, key="go-ask")

        if ask_pressed and question.strip():
            run("ask", compare, question=question.strip())

    with explain_tab:
        pid = st.selectbox("Select Research Paper", list(titles), format_func=titles.get, key="explain-paper")
        focus = st.text_input("Technical Focus (Optional)", placeholder="e.g. loss formulation, dataset preparation, evaluation metrics")
        col_btn2, _ = st.columns([1, 4])
        with col_btn2:
            explain_pressed = st.button("Explain Paper", type="primary", use_container_width=True, key="go-explain")

        if explain_pressed:
            run("explain", compare, paper_key=pid, focus=focus.strip())

    with summary_tab:
        pid2 = st.selectbox("Select Research Paper for Summary", list(titles), format_func=titles.get, key="summary-paper")
        col_btn3, _ = st.columns([1, 4])
        with col_btn3:
            summary_pressed = st.button("Generate Structured Summary", type="primary", use_container_width=True, key="go-summary")

        if summary_pressed:
            run("summarize", compare, paper_key=pid2)
