"""Streamlit front end.   streamlit run app.py

Upload PDFs in the sidebar, then use one of the three modes. Every statement shows its label,
its numbered evidence, and the exact passage behind it; removed claims are listed, not hidden.
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
    page_title="Evidence-Grounded Research Assistant",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Theme & Visual Background Styles
# ---------------------------------------------------------
THEMES = {
    "🌌 Cosmic Aurora": {
        "body_bg": (
            "radial-gradient(circle at 50% -20%, rgba(99, 102, 241, 0.32), transparent 60%), "
            "radial-gradient(circle at 88% 25%, rgba(168, 85, 247, 0.22), transparent 45%), "
            "radial-gradient(circle at 12% 75%, rgba(14, 165, 233, 0.20), transparent 50%), "
            "radial-gradient(circle at 80% 90%, rgba(236, 72, 153, 0.14), transparent 45%), "
            "linear-gradient(180deg, #090d16 0%, #0d1322 50%, #06090e 100%)"
        ),
        "grid_color": "rgba(255, 255, 255, 0.05)",
        "sidebar_bg": "rgba(13, 18, 30, 0.78)",
        "card_bg": "rgba(255, 255, 255, 0.035)",
        "card_hover_border": "rgba(99, 102, 241, 0.5)",
        "border_color": "rgba(255, 255, 255, 0.09)",
        "text_color": "#f1f5f9",
        "muted_color": "#94a3b8",
        "accent": "#6366f1",
        "header_grad": "linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #a5b4fc 100%)",
        "badge_border": "rgba(99, 102, 241, 0.35)",
        "badge_bg": "rgba(99, 102, 241, 0.15)",
        "badge_color": "#a5b4fc",
        "input_bg": "rgba(15, 23, 42, 0.65)",
    },
    "🪐 Deep Space Slate": {
        "body_bg": (
            "radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.15), transparent 65%), "
            "radial-gradient(circle at 90% 80%, rgba(30, 41, 59, 0.4), transparent 50%), "
            "linear-gradient(180deg, #090c15 0%, #0e1424 50%, #070910 100%)"
        ),
        "grid_color": "rgba(255, 255, 255, 0.03)",
        "sidebar_bg": "rgba(10, 14, 26, 0.82)",
        "card_bg": "rgba(255, 255, 255, 0.03)",
        "card_hover_border": "rgba(56, 189, 248, 0.45)",
        "border_color": "rgba(255, 255, 255, 0.07)",
        "text_color": "#f1f5f9",
        "muted_color": "#94a3b8",
        "accent": "#38bdf8",
        "header_grad": "linear-gradient(135deg, #ffffff 0%, #94a3b8 100%)",
        "badge_border": "rgba(56, 189, 248, 0.35)",
        "badge_bg": "rgba(56, 189, 248, 0.15)",
        "badge_color": "#7dd3fc",
        "input_bg": "rgba(14, 20, 36, 0.7)",
    },
    "⚡ Emerald Cyber": {
        "body_bg": (
            "radial-gradient(circle at 80% 10%, rgba(16, 185, 129, 0.24), transparent 45%), "
            "radial-gradient(circle at 10% 80%, rgba(6, 182, 212, 0.20), transparent 45%), "
            "radial-gradient(circle at 50% 50%, rgba(16, 185, 129, 0.08), transparent 60%), "
            "linear-gradient(180deg, #040e0c 0%, #081a17 50%, #030807 100%)"
        ),
        "grid_color": "rgba(52, 211, 153, 0.07)",
        "sidebar_bg": "rgba(6, 17, 15, 0.82)",
        "card_bg": "rgba(16, 185, 129, 0.04)",
        "card_hover_border": "rgba(16, 185, 129, 0.5)",
        "border_color": "rgba(52, 211, 153, 0.15)",
        "text_color": "#ecfdf5",
        "muted_color": "#6ee7b7",
        "accent": "#10b981",
        "header_grad": "linear-gradient(135deg, #ffffff 0%, #a7f3d0 50%, #34d399 100%)",
        "badge_border": "rgba(16, 185, 129, 0.35)",
        "badge_bg": "rgba(16, 185, 129, 0.15)",
        "badge_color": "#6ee7b7",
        "input_bg": "rgba(7, 26, 22, 0.7)",
    },
    "📄 Paper Clean (Light)": {
        "body_bg": (
            "radial-gradient(circle at 50% -10%, rgba(99, 102, 241, 0.1), transparent 60%), "
            "radial-gradient(circle at 85% 75%, rgba(203, 213, 225, 0.5), transparent 45%), "
            "linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)"
        ),
        "grid_color": "rgba(0, 0, 0, 0.04)",
        "sidebar_bg": "rgba(248, 250, 252, 0.88)",
        "card_bg": "rgba(255, 255, 255, 0.85)",
        "card_hover_border": "rgba(99, 102, 241, 0.45)",
        "border_color": "rgba(0, 0, 0, 0.08)",
        "text_color": "#0f172a",
        "muted_color": "#64748b",
        "accent": "#4f46e5",
        "header_grad": "linear-gradient(135deg, #0f172a 0%, #334155 100%)",
        "badge_border": "rgba(79, 70, 229, 0.25)",
        "badge_bg": "rgba(79, 70, 229, 0.1)",
        "badge_color": "#4338ca",
        "input_bg": "#ffffff",
    },
}


def apply_theme_css(theme_name: str) -> None:
    t = THEMES.get(theme_name, THEMES["🌌 Cosmic Aurora"])
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Base */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}

    /* Main Container & Background */
    .stApp {{
        background: {t["body_bg"]} !important;
        background-attachment: fixed !important;
        color: {t["text_color"]} !important;
    }}

    /* High-tech ambient background micro-grid */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background-image: radial-gradient({t["grid_color"]} 1px, transparent 1px);
        background-size: 32px 32px;
        pointer-events: none;
        z-index: 0;
    }}

    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {{
        background: {t["sidebar_bg"]} !important;
        backdrop-filter: blur(18px) saturate(180%);
        -webkit-backdrop-filter: blur(18px) saturate(180%);
        border-right: 1px solid {t["border_color"]} !important;
    }}

    /* Metrics Cards */
    div[data-testid="stMetric"] {{
        background: {t["card_bg"]} !important;
        border: 1px solid {t["border_color"]} !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25) !important;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease !important;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        border-color: {t["card_hover_border"]} !important;
        box-shadow: 0 8px 24px -2px rgba(0, 0, 0, 0.35) !important;
    }}
    div[data-testid="stMetric"] label {{
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: {t["muted_color"]} !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: {t["text_color"]} !important;
    }}

    /* Modern Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {t["card_bg"]};
        padding: 6px;
        border-radius: 12px;
        border: 1px solid {t["border_color"]};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        padding: 8px 18px !important;
        font-weight: 500 !important;
        border: none !important;
        color: {t["muted_color"]} !important;
        transition: all 0.2s ease !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {t["text_color"]} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.22) 0%, rgba(139, 92, 246, 0.22) 100%) !important;
        border: 1px solid {t["accent"]} !important;
        color: {t["text_color"]} !important;
        font-weight: 600 !important;
    }}

    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: {t["card_bg"]} !important;
        border-radius: 10px !important;
        border: 1px solid {t["border_color"]} !important;
        transition: all 0.2s ease !important;
        font-weight: 500 !important;
    }}
    .streamlit-expanderHeader:hover {{
        border-color: {t["accent"]} !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }}

    /* Inputs, Selects & Textareas */
    .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
        background-color: {t["input_bg"]} !important;
        border: 1px solid {t["border_color"]} !important;
        border-radius: 10px !important;
        color: {t["text_color"]} !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }}
    .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: {t["accent"]} !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
    }}

    /* Buttons */
    .stButton button[kind="primary"], .stButton button[type="primary"] {{
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.4rem !important;
        transition: all 0.25s ease !important;
    }}
    .stButton button[kind="primary"]:hover, .stButton button[type="primary"]:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6) !important;
    }}
    .stButton button:not([kind="primary"]):not([type="primary"]) {{
        background: {t["card_bg"]} !important;
        border: 1px solid {t["border_color"]} !important;
        color: {t["text_color"]} !important;
        border-radius: 9px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}
    .stButton button:not([kind="primary"]):not([type="primary"]):hover {{
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: {t["accent"]} !important;
        transform: translateY(-1px) !important;
    }}

    /* Quote Highlight */
    mark {{
        background: linear-gradient(120deg, rgba(245, 158, 11, 0.28) 0%, rgba(234, 179, 8, 0.28) 100%) !important;
        color: #fef08a !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        border-bottom: 2px solid #eab308 !important;
    }}

    /* Code Tags & Marks */
    code {{
        background: rgba(99, 102, 241, 0.15) !important;
        color: #c7d2fe !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 5px !important;
        padding: 1px 6px !important;
        font-size: 0.85em !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header(theme_name: str) -> None:
    t = THEMES.get(theme_name, THEMES["🌌 Cosmic Aurora"])
    st.markdown(
        f"""
        <div style="margin-bottom: 1.75rem; padding: 1.25rem 1.6rem; background: {t['card_bg']}; border: 1px solid {t['border_color']}; border-radius: 16px; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);">
            <div style="display: inline-flex; align-items: center; gap: 8px; padding: 4px 12px; background: {t['badge_bg']}; border: 1px solid {t['badge_border']}; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; color: {t['badge_color']}; margin-bottom: 0.75rem;">
                <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#22c55e; box-shadow:0 0 10px #22c55e;"></span>
                Evidence-Grounded Intelligence • Zero Hallucination
            </div>
            <h1 style="background: {t['header_grad']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 0.5rem 0; font-weight: 800; font-size: 2.2rem; letter-spacing: -0.025em;">
                📑 Evidence-Grounded Research Assistant
            </h1>
            <p style="color: {t['muted_color']}; font-size: 0.95rem; line-height: 1.5; margin: 0; max-width: 880px;">
                Every factual statement is anchored to an exact passage of an uploaded research paper and verified. Statements that fail verification are automatically removed.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


UPLOADS = Path("papers")
COLOR = {"finding": "green", "limitation": "red", "interpretation": "orange"}
NAME = {"finding": "Finding", "limitation": "Limitation", "interpretation": "Interpretation"}


@st.cache_resource(show_spinner="Loading the index and the embedding model...")
def get_engine() -> Engine:
    return Engine(Settings())


def show_answer(ans: Answer) -> None:
    if ans.insufficient and not ans.statements:
        st.warning(NO_ANSWER)
    else:
        ev, ref = numbering(ans)
        by_tag = {p.tag: p for p in ans.passages}
        cols = st.columns(4)
        cols[0].metric("Statements kept", len(ans.statements))
        cols[1].metric("Removed by verification", len(ans.removed))
        cols[2].metric("LLM calls", ans.llm_calls)
        cols[3].metric("Seconds", ans.timings.get("total", 0))
        heading = None
        for st_ in ans.statements:
            if st_.heading and st_.heading != heading:
                heading = st_.heading
                st.markdown(f"#### {heading}")
            marks = " ".join(f"`[{ev[t]}]`" for t in st_.cites if t in ev)
            st.markdown(f":{COLOR[st_.label]}[**{NAME[st_.label]}**]  {st_.text} {marks}")
        st.caption("Finding = stated by the paper.  Limitation = a limit the paper itself states.  "
                   "Interpretation = the system's own reading, not stated by the paper.")
        st.markdown("##### Evidence")
        quotes = {s.tag: s.quote for x in ans.statements for s in x.support if s.quote}
        for tag, n in ev.items():
            p = by_tag[tag]
            with st.expander(f"[{n}]  {p.paper.short()} · {p.chunk.pages} · {p.chunk.section or 'n/a'}"):
                text, quote = html.escape(p.chunk.text), html.escape(quotes.get(tag, ""))
                if quote and quote in text:
                    text = text.replace(quote, f"<mark>{quote}</mark>")
                st.markdown(text, unsafe_allow_html=True)
                st.caption(f"{p.paper.title} · chunk {p.chunk.chunk_id}")
        st.markdown("##### References")
        papers = {p.paper.paper_id: p.paper for p in ans.passages}
        for pid, n in ref.items():
            st.markdown(f"{n}. {ieee_reference(papers[pid])}")
    if ans.removed:
        with st.expander(f"Removed by verification ({len(ans.removed)})"):
            for s in ans.removed:
                st.markdown(f"- ~~{s.text}~~  \n  _{NAME[s.label]} · {s.reason}_")


def show_baseline(out: dict) -> None:
    st.caption("Plain LLM, no documents. Its references are written from memory and are not checked.")
    for s in out["statements"]:
        st.markdown(f"- {s['text']} " + " ".join(f"`[{n}]`" for n in s["refs"]))
    for r in out["references"]:
        st.markdown(f"{r['n']}. {r['authors']}, \u201c{r['title']},\u201d {r['venue']}, {r['year'] or 'n.d.'}.")


def run(mode: str, compare: bool, **kw) -> None:
    engine = get_engine()
    try:
        with st.spinner("Retrieving, generating, verifying..."):
            ans = {"ask": engine.ask, "explain": engine.explain, "summarize": engine.summarize}[mode](**kw)
        if compare:
            left, right = st.columns(2)
            with left:
                st.subheader("Evidence-grounded")
                show_answer(ans)
            with right:
                st.subheader("Plain LLM")
                with st.spinner("Asking the plain LLM..."):
                    show_baseline(engine.baseline(mode, question=kw.get("question", ""), paper_key=kw.get("paper_key", "")))
        else:
            show_answer(ans)
    except Exception as exc:  # shown to the user instead of a stack trace
        st.error(f"{type(exc).__name__}: {exc}")


engine = get_engine()
with st.sidebar:
    st.markdown("### 🎨 Appearance")
    bg_theme = st.selectbox(
        "Background Theme",
        list(THEMES.keys()),
        index=0,
        help="Select a background visual style for the workspace.",
    )
    apply_theme_css(bg_theme)
    st.markdown("---")
    st.header("Papers")
    files = st.file_uploader("Add research papers (PDF)", type="pdf", accept_multiple_files=True)
    enrich = st.checkbox("Look up title, authors and year online (Crossref / arXiv)", value=False)
    if files and st.button("Ingest", type="primary"):
        UPLOADS.mkdir(exist_ok=True)
        for f in files:
            path = UPLOADS / f.name
            path.write_bytes(f.getbuffer())
            try:
                paper = engine.ingest(path, enrich=enrich)
                st.success(f"{paper.title[:60]} ({paper.n_chunks} chunks)")
            except Exception as exc:
                st.error(f"{f.name}: {exc}")
    for paper in engine.papers():
        with st.expander(paper.title[:70]):
            st.write(f"**Authors:** {', '.join(paper.authors) or 'not found'}")
            st.write(f"**Year:** {paper.year or 'not found'}  \n**DOI:** {paper.doi or 'not found'}")
            st.caption(f"{paper.n_pages} pages · {paper.n_chunks} chunks · metadata from {paper.meta_source} · id {paper.paper_id}")
            if st.button("Remove", key=f"rm-{paper.paper_id}"):
                engine.store.remove_paper(paper.paper_id)
                st.rerun()
    compare = st.checkbox("Also show the plain LLM answer", value=False)

render_header(bg_theme)
papers = engine.papers()
if not papers:
    st.info("Add one or more research papers in the sidebar to begin.")
else:
    titles = {p.paper_id: f"{p.title[:90]} ({p.short()})" for p in papers}
    ask_tab, explain_tab, summary_tab = st.tabs(["Ask Research", "Explain Paper", "Summarize Paper"])
    with ask_tab:
        question = st.text_area("Question across all uploaded papers", height=80)
        if st.button("Answer", key="go-ask") and question.strip():
            run("ask", compare, question=question.strip())
    with explain_tab:
        pid = st.selectbox("Paper", list(titles), format_func=titles.get, key="explain-paper")
        focus = st.text_input("Focus (optional)", placeholder="e.g. the loss function, the experimental setup")
        if st.button("Explain", key="go-explain"):
            run("explain", compare, paper_key=pid, focus=focus.strip())
    with summary_tab:
        pid2 = st.selectbox("Paper", list(titles), format_func=titles.get, key="summary-paper")
        if st.button("Summarize", key="go-summary"):
            run("summarize", compare, paper_key=pid2)
