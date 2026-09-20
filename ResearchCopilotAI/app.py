"""
app.py
------
Research Paper Co-Pilot — Verifiable Multimodal Scientific Assistant & IEEE Studio.

Color Palette (Derived from User Specification):
- #302F38: Deep Charcoal / Main Contrast Text
- #2F3A6E: Deep Indigo / Navy Blue (Primary Accent, Header Gradient, Primary Buttons)
- #7174B9: Soft Lavender / Periwinkle (Active Tabs, Links, Hover Accent)
- #ABC4E6: Ice Blue / Soft Periwinkle (Source Chips, Highlight Cards, Citation Surfaces)
- #C0CDEC: Very Light Periwinkle / Surface Background Accent
- #F5F7FC: Main Page Background (Modern Ultra-Clean Light Mode)
"""

import streamlit as st
import pandas as pd

from config import (
    APP_TITLE, APP_ICON
)
from utils.pdf_loader import save_uploaded_pdf
from utils.multimodal_parser import parse_multimodal_pdf, chunk_parsed_document
from utils.embeddings import add_documents_to_vector_store, get_all_paper_names, delete_paper_from_vector_store
from utils.hybrid_retriever import hybrid_retriever
from utils.orchestrator import orchestrator
from utils.llm import generate_response
from utils.paper_analysis import analyze_paper
from utils.literature_review import generate_literature_review
from utils.research_gap import detect_research_gaps
from utils.conflict_detector import detect_cross_paper_conflicts
from utils.formula_extractor import extract_formulas_and_metrics

# ==========================================================================
# PAGE CONFIGURATION & MODERN MINIMALIST DESIGN SYSTEM
# ==========================================================================
st.set_page_config(
    page_title=f"{APP_TITLE} | Verifiable Multi-Paper AI Suite",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #14323a;
        --muted: #5d747b;
        --line: #cce5e9;
        --page: #f7fbfc;
        --panel: #ffffff;
        --surface-muted: #eaf7f9;
        --heading: #14323a;
        --heading-accent: #32bacd;
        --primary: #168fa5;
        --primary-dark: #0e6678;
        --accent: #32bacd;
        --warning: #D97706;
        --danger: #DC2626;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(circle at 90% 0%, rgba(50,186,205,.12), transparent 28rem),
            var(--page);
    }

    .block-container {
        max-width: 1380px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, h5, h6 {
        letter-spacing: 0 !important;
        color: var(--ink) !important;
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"] {
        color: var(--heading);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3f4a 0%, #0d5968 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
        min-width: 290px;
    }

    section[data-testid="stSidebar"] * {
        color: #e9fbfd;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaptionContainer {
        color: #b8dce2 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(213,247,250,0.22);
        margin: 1rem 0;
    }

    /* Dropdown Popover List Max-Height & Custom Scrollbar Fix */
    div[data-baseweb="popover"] div[role="listbox"],
    ul[role="listbox"],
    div[data-baseweb="menu"],
    div[data-baseweb="select"] ul {
        max-height: 260px !important;
        overflow-y: auto !important;
        border-radius: 10px !important;
        border: 1px solid var(--line) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25) !important;
    }
    div[data-baseweb="popover"] li[role="option"] {
        padding: 10px 14px !important;
        font-size: 13.5px !important;
    }
    div[data-baseweb="popover"] ::-webkit-scrollbar {
        width: 6px !important;
    }
    div[data-baseweb="popover"] ::-webkit-scrollbar-thumb {
        background: var(--heading-accent) !important;
        border-radius: 999px !important;
    }
    div[data-baseweb="popover"] ::-webkit-scrollbar-track {
        background: var(--surface-muted) !important;
    }


    .sidebar-brand {
        border: 1px solid rgba(255,255,255,0.10);
        background: transparent;
        border-bottom: 1px solid rgba(213,247,250,0.22);
        border-radius: 0;
        padding: 4px 2px 20px;
        margin-bottom: 22px;
    }

    .sidebar-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 22px;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.2;
        margin-bottom: 6px;
    }

    .sidebar-subtitle {
        font-size: 12px;
        color: #b8dce2;
        line-height: 1.45;
    }

    .sidebar-panel {
        border: 0;
        border-top: 1px solid rgba(213,247,250,0.22);
        background: transparent;
        border-radius: 0;
        padding: 18px 2px 4px;
        margin: 18px 0;
    }

    .sidebar-panel-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        font-size: 12px;
        text-transform: uppercase;
        color: #b9edf2;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero-banner-container {
        background: transparent;
        border-bottom: 1px solid var(--line);
        border-radius: 0;
        padding: 0 0 24px;
        margin-bottom: 22px;
        box-shadow: none;
    }

    .hero-topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 18px;
        flex-wrap: wrap;
    }

    .hero-kicker {
        color: var(--primary);
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
    }

    .hero-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #e8f4ef;
        color: var(--primary-dark);
        border: 1px solid #b9ded0;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 700;
    }

    .hero-title {
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(30px, 4vw, 46px);
        line-height: 1.08;
        font-weight: 800;
        margin: 0 0 10px 0;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 15px;
        max-width: 820px;
        line-height: 1.65;
        margin: 0;
    }

    .scope-pill {
        background: #edf9fa;
        border: 1px solid var(--line);
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: 14px 16px;
        margin: 4px 0 18px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        color: var(--ink);
        box-shadow: 0 8px 24px rgba(16, 24, 40, 0.045);
    }

    .scope-title {
        font-size: 13px;
        font-weight: 800;
        color: var(--ink);
    }

    .scope-detail {
        font-size: 12px;
        color: var(--muted);
        margin-top: 4px;
    }

    .scope-count {
        flex: 0 0 auto;
        background: #e1f5f8;
        color: var(--primary-dark);
        border: 1px solid #a9dfe7;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 12px;
        font-weight: 800;
    }

    .modern-header {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 24px;
        font-weight: 800;
        color: var(--heading);
        margin: 10px 0 12px 0;
        padding-bottom: 10px;
        border-bottom: 0;
    }

    .modern-header::before {
        content: "";
        width: 8px;
        height: 26px;
        border-radius: 999px;
        background: var(--heading-accent);
        display: inline-block;
    }

    .heading-ask::before { background: #ddf0f5; }
    .heading-teardown::before { background: #b2dee6; }
    .heading-conflicts::before { background: #85ceda; }
    .heading-formulas::before { background: #32bacd; }
    .heading-review::before { background: #20aec2; }
    .heading-draft::before { background: #18a3b6; }

    .section-note {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.6;
        margin: -4px 0 18px 20px;
    }

    .modern-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 8px 20px rgba(29, 48, 43, 0.045);
        color: var(--ink);
    }

    .result-panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 20px;
        margin: 14px 0;
    }

    .citation-quote {
        border-left: 3px solid var(--primary);
        background: var(--surface-muted);
        padding: 12px 14px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
        color: var(--heading);
        font-size: 13px;
        line-height: 1.6;
    }

    .badge-chip, .badge-conflict {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .02em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .badge-chip {
        background: #e1f5f8;
        color: var(--primary-dark);
        border: 1px solid #a9dfe7;
    }

    .badge-conflict {
        background: #FEF3F2;
        color: #B42318;
        border: 1px solid #FECDCA;
    }

    div[data-baseweb="tab-list"] {
        gap: 10px !important;
        background: transparent !important;
        padding: 6px 0 10px !important;
        border-radius: 0 !important;
        border: 0 !important;
        margin: 10px 0 22px 0 !important;
        box-shadow: none;
        overflow-x: auto !important;
    }

    button[data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 12px 16px !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        color: #49666e !important;
        background: #ffffff !important;
        border: 1px solid var(--line) !important;
        box-shadow: 0 5px 14px rgba(14, 102, 120, 0.06) !important;
        white-space: nowrap !important;
    }

    button[data-baseweb="tab"]:hover {
        background: #e1f5f8 !important;
        color: var(--ink) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: #168fa5 !important;
        color: #FFFFFF !important;
        border-color: #168fa5 !important;
        box-shadow: 0 8px 18px rgba(22, 143, 165, 0.24) !important;
    }

    button[data-baseweb="tab"]:nth-child(1) { border-top: 3px solid #ddf0f5 !important; }
    button[data-baseweb="tab"]:nth-child(2) { border-top: 3px solid #b2dee6 !important; }
    button[data-baseweb="tab"]:nth-child(3) { border-top: 3px solid #85ceda !important; }
    button[data-baseweb="tab"]:nth-child(4) { border-top: 3px solid #32bacd !important; }
    button[data-baseweb="tab"]:nth-child(5) { border-top: 3px solid #20aec2 !important; }
    button[data-baseweb="tab"]:nth-child(6) { border-top: 3px solid #18a3b6 !important; }

    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
        display: none !important;
    }

    div.stButton > button {
        border-radius: 6px !important;
        border: 1px solid #C7D7FE !important;
        background: #FFFFFF !important;
        color: var(--primary-dark) !important;
        font-weight: 800 !important;
        padding: 0.62rem 1rem !important;
        box-shadow: 0 5px 12px rgba(29, 48, 43, 0.08) !important;
    }

    div.stButton > button[kind="primary"] {
        background: var(--primary) !important;
        color: #FFFFFF !important;
        border-color: var(--primary) !important;
    }

    div.stButton > button:hover {
        border-color: var(--primary) !important;
        transform: translateY(-1px);
    }

    div[data-testid="stFileUploader"], div[data-testid="stSelectbox"], div[data-testid="stMultiSelect"], div[data-testid="stRadio"] {
        border-radius: 8px;
    }

    div[data-testid="stExpander"] {
        background: var(--panel) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 6px 16px rgba(16, 24, 40, 0.035) !important;
        overflow: hidden !important;
    }

    div[data-testid="stExpander"] summary {
        font-weight: 800 !important;
        color: var(--ink) !important;
    }

    .stApp table {
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    .stApp th {
        background: var(--surface-muted) !important;
        color: var(--heading) !important;
        font-weight: 800 !important;
    }

    .app-footer-container {
        background: transparent;
        border-top: 1px solid var(--line);
        border-radius: 0;
        padding: 16px 0 4px;
        color: var(--muted);
        margin: 42px 0 0;
    }

    .footer-row {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 18px;
        flex-wrap: wrap;
    }

    .footer-brand {
        font-size: 14px;
        font-weight: 800;
        color: var(--primary-dark);
    }

    .footer-sub {
        color: var(--muted);
        font-size: 12px;
        margin-top: 6px;
        max-width: 620px;
        line-height: 1.6;
    }

    .footer-copy {
        color: var(--muted);
        font-size: 12px;
        border-top: 0;
        padding-top: 0;
        margin-top: 8px;
    }

    @media (max-width: 900px) {
        .scope-pill {
            align-items: flex-start;
            flex-direction: column;
        }
        .block-container { padding: 1.25rem 1rem 2rem; }
        .hero-title { font-size: 32px; }
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --ink: #e8f8fa;
            --muted: #a8cbd0;
            --line: #2c555e;
            --page: #0b171a;
            --panel: #13272c;
            --surface-muted: #18383f;
            --heading: #e8f8fa;
            --heading-accent: #32bacd;
            --primary: #32bacd;
            --primary-dark: #b9edf2;
            --accent: #32bacd;
        }
        .stApp { background: radial-gradient(circle at 90% 0%, rgba(50,186,205,.14), transparent 28rem), var(--page); }
        .hero-status, .scope-pill { background: #123b43; color: #d6f5f7; }
        .scope-pill, .modern-card, .result-panel { border-color: var(--line); }
        .scope-count, .badge-chip { background: #154650; color: #d2f5f8; border-color: #287381; }
        .citation-quote { color: #d4edf0; }
        div.stButton > button { background: var(--panel) !important; color: #d6f5f7 !important; border-color: #397985 !important; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_pipeline_result" not in st.session_state:
    st.session_state.last_pipeline_result = None

# Fetch paper list from database/uploads
paper_names = get_all_paper_names()

# ==========================================================================
# SIDEBAR REPOSITORY & MULTI-PAPER SELECTION CONTROLS
# ==========================================================================
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-title">ResearchX</div>
            <div class="sidebar-subtitle">Multi-Agent Research Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-panel"><div class="sidebar-panel-title"><span>Add sources</span><span>PDF</span></div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload academic papers to compare layout, methodology, formulas, and empirical results side-by-side."
    )
    
    if uploaded_files:
        if st.button("Add to workspace", use_container_width=True, type="primary"):
            with st.spinner("Parsing layout, extracting sections, and indexing paper..."):
                last_name = None
                for up_file in uploaded_files:
                    save_path = save_uploaded_pdf(up_file)
                    parsed_doc = parse_multimodal_pdf(save_path)
                    chunks = chunk_parsed_document(parsed_doc)
                    from langchain_core.documents import Document
                    docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]
                    add_documents_to_vector_store(docs)
                    last_name = parsed_doc["paper_name"]
                
                hybrid_retriever.sync_bm25_from_vector_store()
                st.success("Indexed uploaded paper(s) successfully!")
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-panel"><div class="sidebar-panel-title"><span>Choose focus</span><span>View</span></div>', unsafe_allow_html=True)
    if paper_names:
        scope_mode = st.radio(
            "Choose workflow",
            ["Compare multiple papers", "Single paper deep-dive"],
            index=0 if len(paper_names) > 1 else 1,
            help="Choose whether to compare multiple research papers side-by-side or focus on 1 paper."
        )

        if scope_mode == "Compare multiple papers":
            selected_papers = st.multiselect(
                "Papers to compare",
                options=paper_names,
                default=paper_names,
                help="Select 2 or more research papers to compare side-by-side."
            )
            if not selected_papers:
                selected_papers = paper_names
            target_papers = selected_papers
            st.session_state.active_paper = "Comparative reading"
            selected_filter = None
        else:
            selected_single = st.selectbox(
                "Target paper",
                options=paper_names,
                index=0,
                help="Analysis will focus strictly on this single document."
            )
            target_papers = [selected_single]
            st.session_state.active_paper = selected_single
    else:
        target_papers = []
        selected_filter = None
        st.info("No papers added yet. Upload PDF paper(s) above to begin.")
    st.markdown("</div>", unsafe_allow_html=True)


    if paper_names:
        st.markdown('<div class="sidebar-panel"><div class="sidebar-panel-title"><span>Manage papers</span><span>Delete</span></div>', unsafe_allow_html=True)
        with st.expander("🗑️ Delete Previous Papers", expanded=False):
            st.caption("Permanently remove selected paper(s) from storage & Chroma vector database.")
            papers_to_del = st.multiselect(
                "Select paper(s) to remove",
                options=paper_names,
                key="del_papers_multiselect",
                help="Selected paper(s) will be permanently deleted from database and disk."
            )
            if st.button("🔴 Permanently Delete Selected Paper(s)", use_container_width=True):
                if papers_to_del:
                    for p_del in papers_to_del:
                        delete_paper_from_vector_store(p_del)
                    hybrid_retriever.sync_bm25_from_vector_store()
                    st.success(f"Permanently removed {len(papers_to_del)} paper(s)!")
                    st.rerun()
                else:
                    st.warning("Please select at least one paper to delete.")
        st.markdown("</div>", unsafe_allow_html=True)


# Show only the active reading context, without exposing system metrics.
def render_scope_banner():
    if paper_names and target_papers:
        if len(target_papers) > 1:
            display_names = ", ".join([p[:34] + ("..." if len(p) > 34 else "") for p in target_papers[:3]])
            if len(target_papers) > 3:
                display_names += f" + {len(target_papers) - 3} more"
            st.markdown(
                f"""
                <div class="scope-pill">
                    <div>
                        <div class="scope-title">Comparative reading</div>
                        <div class="scope-detail">{display_names}</div>
                    </div>
                        <div class="scope-count">Selected sources</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            p_name = target_papers[0]
            st.markdown(
                f"""
                <div class="scope-pill">
                    <div>
                        <div class="scope-title">Focused reading</div>
                        <div class="scope-detail">{p_name}</div>
                    </div>
                        <div class="scope-count">Active source</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("No paper loaded in repository. Upload PDF research paper(s) using the sidebar to begin analysis.")

# Top workspace header
st.markdown(f"""
<div class="hero-banner-container">
    <div class="hero-topline">
        <div class="hero-kicker">Research workspace</div>
        <div class="hero-status">Ready to explore</div>
    </div>
    <h1 class="hero-title">Make sense of your research.</h1>
    <p class="hero-subtitle">Read evidence, compare ideas, uncover gaps, and turn your notes into a clear academic draft.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================================
# MAIN PRODUCT TABS
# ==========================================================================
tabs = st.tabs([
    "Ask",
    "Understand",
    "Compare",
    "Measure",
    "Discover",
    "Write"
])

# --------------------------------------------------------------------------
# TAB 1: GROUNDED AI RESEARCH ASSISTANT (CONTINUOUS CHAT)
# --------------------------------------------------------------------------
with tabs[0]:
    st.markdown("<div class='modern-header heading-ask'>Ask your papers</div>", unsafe_allow_html=True)
    render_scope_banner()
    st.markdown("<div class='section-note'>Ask follow-up questions across the selected papers. Answers are grounded in retrieved source passages.</div>", unsafe_allow_html=True)

    # Render Chat History Feed
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"**Question ({entry.get('paper', 'Target')}):** {entry['question']}")
        with st.chat_message("assistant"):
            st.markdown(f"<span class='badge-chip'>VERIFIED CITATION GROUNDING</span>", unsafe_allow_html=True)
            st.markdown(entry['answer'])
            with st.expander("View source passages and citation details", expanded=False):
                for idx, src in enumerate(entry.get("sources", [])):
                    meta = src.get("metadata", {})
                    st.markdown(f"**[{idx+1}] {meta.get('paper_name')}** - Page {meta.get('page_number')} (Section: {meta.get('section', 'General')})")
                    st.markdown(f"<div class='citation-quote'>\"{src.get('text')[:280]}...\"</div>", unsafe_allow_html=True)

    # Continuous Chat Input
    if prompt := st.chat_input("Ask a question comparing your selected papers...", disabled=not paper_names):
        with st.chat_message("user"):
            st.markdown(f"**Question ({st.session_state.active_paper}):** {prompt}")

        with st.chat_message("assistant"):
            with st.spinner(f"Retrieving passages across {len(target_papers)} paper(s)..."):
                retrieved_chunks = []
                for p_target in target_papers:
                    retrieved_chunks.extend(hybrid_retriever.retrieve(
                        query=prompt,
                        top_k=3,
                        paper_name=p_target
                    ))
                
                context_str = "\n\n".join([
                    f"[Source: {c.get('metadata', {}).get('paper_name')} | Page: {c.get('metadata', {}).get('page_number')}]\n{c.get('text')}"
                    for c in retrieved_chunks
                ])

                chat_prompt = f"""You are an expert academic research assistant comparing research literature.
Answer the user's question using ONLY the provided literature context for the selected papers: {', '.join(target_papers)}.
Include bracketed citations [PaperName, p.X] for every key fact.

Literature Context:
{context_str}

User Question: {prompt}
"""
                ai_answer = generate_response(chat_prompt)
                st.markdown(f"<span class='badge-chip'>VERIFIED CITATION GROUNDING</span>", unsafe_allow_html=True)
                st.markdown(ai_answer)

                with st.expander("View source passages and citation details", expanded=False):
                    for idx, src in enumerate(retrieved_chunks):
                        meta = src.get("metadata", {})
                        st.markdown(f"**[{idx+1}] {meta.get('paper_name')}** - Page {meta.get('page_number')} (Section: {meta.get('section', 'General')})")
                        st.markdown(f"<div class='citation-quote'>\"{src.get('text')[:280]}...\"</div>", unsafe_allow_html=True)

                st.session_state.chat_history.append({
                    "paper": st.session_state.active_paper,
                    "question": prompt,
                    "answer": ai_answer,
                    "sources": retrieved_chunks
                })

# --------------------------------------------------------------------------
# TAB 2: DOCUMENT ANALYSIS TEARDOWN
# --------------------------------------------------------------------------
with tabs[1]:
    st.markdown("<div class='modern-header heading-teardown'>Understand a paper</div>", unsafe_allow_html=True)
    render_scope_banner()

    if target_papers:
        if st.button(f"Generate teardown breakdown ({len(target_papers)} selected papers)", type="primary", use_container_width=True):
            with st.spinner("Extracting structured academic breakdowns for selected papers..."):
                t_cols = st.tabs([f"{p[:30]}..." for p in target_papers])
                for idx, p_name in enumerate(target_papers):
                    with t_cols[idx]:
                        analysis_res = analyze_paper(p_name)
                        st.markdown(f"### Academic Breakdown: `{p_name}`")
                        for field, content in analysis_res.items():
                            with st.expander(str(field), expanded=True):
                                st.write(content)

# --------------------------------------------------------------------------
# TAB 3: CROSS-PAPER CONFLICT DETECTOR
# --------------------------------------------------------------------------
with tabs[2]:
    st.markdown("<div class='modern-header heading-conflicts'>Compare findings</div>", unsafe_allow_html=True)
    render_scope_banner()
    st.markdown("<div class='section-note'>Find contradictions, methodological trade-offs, and reasons results differ across papers.</div>", unsafe_allow_html=True)

    if target_papers:
        if st.button(f"Run conflict analysis ({len(target_papers)} selected papers)", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing cross-paper refutation & empirical variance for {st.session_state.active_paper}..."):
                conflict_res = detect_cross_paper_conflicts(target_papers)
                st.markdown(f"### Contradiction & Variance Analysis ({st.session_state.active_paper})")
                st.write(conflict_res.get("summary", ""))

                for idx, c in enumerate(conflict_res.get("conflicts", [])):
                    with st.container():
                        st.markdown(f"#### <span class='badge-conflict'>CONFLICT / TRADE-OFF #{idx+1}</span> {c.get('conflict_topic')}", unsafe_allow_html=True)
                        col_f1, col_f2 = st.columns(2)
                        with col_f1:
                            st.markdown(f"<div class='modern-card'><b>Perspective / Claim A:</b><br>{c.get('paper_a_claim')}</div>", unsafe_allow_html=True)
                        with col_f2:
                            st.markdown(f"<div class='modern-card'><b>Opposing Claim / Risk B:</b><br>{c.get('paper_b_claim')}</div>", unsafe_allow_html=True)
                        st.info(f"**Root Cause of Variance:** {c.get('root_cause')}")
                        st.success(f"**Reconciliation Hypothesis:** {c.get('reconciliation_hypothesis')}")
                        st.markdown("---")

# --------------------------------------------------------------------------
# TAB 4: FORMULAS & QUANTITATIVE SETUP
# --------------------------------------------------------------------------
with tabs[3]:
    st.markdown("<div class='modern-header heading-formulas'>Measure the details</div>", unsafe_allow_html=True)
    render_scope_banner()
    st.markdown("<div class='section-note'>Extract equations, parameters, datasets, benchmark values, and experimental setup details.</div>", unsafe_allow_html=True)

    if target_papers:
        if st.button(f"Extract quantitative setup ({len(target_papers)} selected papers)", type="primary", use_container_width=True):
            with st.spinner("Extracting quantitative formulations and metrics across papers..."):
                f_tabs = st.tabs([f"{p[:30]}..." for p in target_papers])
                for idx, p_name in enumerate(target_papers):
                    with f_tabs[idx]:
                        f_res = extract_formulas_and_metrics(p_name)
                        st.markdown(f"### Quantitative Setup: `{p_name}`")
                        st.write(f_res.get("summary", ""))

                        st.markdown("#### Extracted Equations & Formulas")
                        eqs = f_res.get("equations", [])
                        if eqs:
                            for eq in eqs:
                                with st.expander(f"Equation: {eq.get('name')}", expanded=True):
                                    st.latex(eq.get("latex", ""))
                                    st.caption(eq.get("description", ""))
                        else:
                            st.info("No formal equations found in document text.")

                        st.markdown("#### Hardware Sensors, Parameters & Dataset Metrics")
                        hp_data = f_res.get("hyperparameters_and_setup", [])
                        if hp_data:
                            st.dataframe(pd.DataFrame(hp_data), use_container_width=True)

# --------------------------------------------------------------------------
# TAB 5: LITERATURE SYNTHESIS & RESEARCH GAPS
# --------------------------------------------------------------------------
with tabs[4]:
    st.markdown("<div class='modern-header heading-review'>Discover new directions</div>", unsafe_allow_html=True)
    render_scope_banner()

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("#### Literature Review Synthesis")
        if st.button(f"Synthesize literature review ({len(target_papers)} papers)", type="primary", use_container_width=True, disabled=not target_papers):
            with st.spinner(f"Synthesizing comparative review for {len(target_papers)} paper(s)..."):
                rev = generate_literature_review(target_papers)
                for k, v in rev.items():
                    with st.expander(str(k), expanded=True):
                        st.write(v)

    with col_r2:
        st.markdown("#### Research Gap Matrix")
        if st.button(f"Discover cross-paper research gaps ({len(target_papers)} papers)", type="primary", use_container_width=True, disabled=not target_papers):
            with st.spinner(f"Discovering cross-paper research gaps for {len(target_papers)} paper(s)..."):
                gaps = detect_research_gaps(target_papers)
                for k, v in gaps.items():
                    with st.expander(str(k), expanded=True):
                        st.info(v)

# --------------------------------------------------------------------------
# TAB 6: IEEE MANUSCRIPT & LATEX STUDIO
# --------------------------------------------------------------------------
with tabs[5]:
    st.markdown("<div class='modern-header heading-draft'>Write your draft</div>", unsafe_allow_html=True)
    render_scope_banner()

    paper_topic = st.text_input(
        "IEEE Paper Focus / Title:",
        value=f"A Comparative Survey and Verifiable Analysis of {len(target_papers)} Academic Literature Frameworks"
    )
    
    if st.button("Generate multi-paper IEEE draft and LaTeX", type="primary", disabled=not target_papers, use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(msg: str, pct: int):
            status_text.markdown(f"**{msg}**")
            progress_bar.progress(pct)

        with st.spinner(f"Orchestrating 6-agent pipeline for IEEE manuscript synthesis across {len(target_papers)} paper(s)..."):
            pipe_res = orchestrator.run_full_pipeline(
                research_query=paper_topic,
                selected_paper=selected_filter,
                progress_callback=update_progress
            )
            st.session_state.last_pipeline_result = pipe_res
            st.success("IEEE Comparative Conference Draft & LaTeX generated successfully!")

    res = st.session_state.last_pipeline_result
    if res and "agent6_output" in res:
        a6 = res["agent6_output"]
        sec = a6.get("sections", {})
        kw_str = ", ".join(a6.get("keywords", []))

        st.markdown(
            f"""
            <div class="result-panel">
                <span class="badge-chip">IEEE manuscript preview</span>
                <h2>{a6.get('paper_title')}</h2>
                <p><b>Abstract</b> - {a6.get('abstract')}</p>
                <p><b>Index Terms</b> - {kw_str}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        for s_name, s_title in [
            ("introduction", "I. INTRODUCTION"),
            ("related_work", "II. RELATED WORK & COMPARATIVE TAXONOMY"),
            ("methodology", "III. SYSTEM ARCHITECTURE & COMPARATIVE FRAMEWORK"),
            ("experiments_and_results", "IV. EXPERIMENTAL COMPARISON & EMPIRICAL RESULTS"),
            ("discussion_and_gaps", "V. DISCUSSION & CROSS-PAPER RESEARCH GAPS"),
            ("conclusion", "VI. CONCLUSION"),
        ]:
            st.markdown(f"### {s_title}")
            st.write(sec.get(s_name, ""))

        st.markdown("### REFERENCES")
        ref_list = [f"- {r}" for r in a6.get("ieee_references", [])]
        for r in ref_list:
            st.markdown(r)

        st.markdown("### BibTeX Entries")
        st.code(a6.get("bibtex_entries", ""), language="bibtex")

        st.markdown("### IEEE LaTeX Source Code (`.tex`)")
        st.code(a6.get("latex_source", "% IEEE LaTeX source"), language="latex")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            formatted_refs = "\n".join(ref_list)
            full_md = f"# {a6.get('paper_title')}\n\n**Abstract** - {a6.get('abstract')}\n\n**Keywords** - {kw_str}\n\n## I. INTRODUCTION\n{sec.get('introduction')}\n\n## II. RELATED WORK & COMPARATIVE TAXONOMY\n{sec.get('related_work')}\n\n## III. SYSTEM ARCHITECTURE & COMPARATIVE FRAMEWORK\n{sec.get('methodology')}\n\n## IV. EXPERIMENTAL COMPARISON & EMPIRICAL RESULTS\n{sec.get('experiments_and_results')}\n\n## V. DISCUSSION & CROSS-PAPER RESEARCH GAPS\n{sec.get('discussion_and_gaps')}\n\n## VI. CONCLUSION\n{sec.get('conclusion')}\n\n## REFERENCES\n{formatted_refs}\n\n## BIBTEX\n```bibtex\n{a6.get('bibtex_entries')}\n```\n"
            st.download_button("Download Markdown draft (.md)", data=full_md, file_name="IEEE_Comparative_Manuscript.md", mime="text/markdown", use_container_width=True)
        with col_d2:
            st.download_button("Download Overleaf LaTeX (.tex)", data=a6.get("latex_source", ""), file_name="IEEE_Comparative_Manuscript.tex", mime="text/x-tex", use_container_width=True)

# ==========================================================================
# MODERN PRODUCT FOOTER
# ==========================================================================
st.markdown("""
<div class="app-footer-container">
    <div class="footer-row">
        <div>
            <div class="footer-brand">ResearchX — Multi-Agent Research Intelligence</div>
            <div class="footer-sub">Read clearly. Think deeply. Write confidently.</div>
        </div>
    </div>
    <div class="footer-copy clean-footer-copy">Your academic reading workspace</div>
</div>
""", unsafe_allow_html=True)
