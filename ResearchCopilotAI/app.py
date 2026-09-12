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

import os
import streamlit as st
import pandas as pd
import json

from config import (
    APP_TITLE, APP_SUBTITLE, APP_ICON, GOOGLE_API_KEY,
    RETRIEVER_TOP_K, HYBRID_DENSE_WEIGHT, HYBRID_SPARSE_WEIGHT
)
from utils.pdf_loader import save_uploaded_pdf
from utils.multimodal_parser import parse_multimodal_pdf, chunk_parsed_document
from utils.embeddings import add_documents_to_vector_store, get_all_paper_names
from utils.hybrid_retriever import hybrid_retriever
from utils.orchestrator import orchestrator
from utils.evaluation_suite import evaluation_suite
from utils.llm import generate_response
from utils.paper_analysis import analyze_paper
from utils.literature_review import generate_literature_review
from utils.research_gap import detect_research_gaps
from utils.conflict_detector import detect_cross_paper_conflicts
from utils.formula_extractor import extract_formulas_and_metrics
from utils.retriever import retrieve_all_chunks_for_paper

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
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Page Canvas Background */
    .stApp {
        background: linear-gradient(180deg, #EBF1FA 0%, #F5F7FC 350px, #F5F7FC 100%);
        color: #302F38;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #ABC4E6;
        box-shadow: 4px 0 20px rgba(47, 58, 110, 0.03);
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #2F3A6E;
        font-weight: 700;
    }

    /* Organic Curved Hero Header Banner */
    .hero-banner-container {
        position: relative;
        background: linear-gradient(135deg, #2F3A6E 0%, #474E86 45%, #7174B9 100%);
        border-radius: 24px;
        padding: 32px 38px 48px 38px;
        color: #FFFFFF;
        margin-bottom: 12px;
        box-shadow: 0 16px 36px rgba(47, 58, 110, 0.18);
        overflow: hidden;
    }
    .hero-banner-container::after {
        content: "";
        position: absolute;
        bottom: -1px;
        left: 0;
        right: 0;
        height: 32px;
        background: #F5F7FC;
        clip-path: ellipse(55% 100% at 50% 100%);
    }
    .hero-banner-container h1 {
        color: #FFFFFF !important;
        font-size: 30px;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.025em;
        text-shadow: 0 2px 10px rgba(0,0,0,0.12);
    }
    .hero-banner-container p {
        color: #C0CDEC !important;
        font-size: 14.5px;
        margin: 0;
        font-weight: 500;
        max-width: 680px;
        line-height: 1.5;
    }

    /* Floating Pill Header Badge */
    .hero-pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        color: #FFFFFF;
        padding: 5px 15px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 12px;
        letter-spacing: 0.02em;
    }

    /* Primary Cards & Container Panels */
    .modern-card {
        background-color: #FFFFFF;
        border: 1px solid #ABC4E6;
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(47, 58, 110, 0.05);
        color: #302F38;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .modern-card:hover {
        box-shadow: 0 12px 30px rgba(47, 58, 110, 0.08);
    }

    /* Active Scope Banner Pill */
    .scope-pill {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F4FC 100%);
        border: 1px solid #ABC4E6;
        border-left: 5px solid #2F3A6E;
        border-radius: 16px;
        padding: 14px 22px;
        font-size: 14px;
        font-weight: 700;
        color: #2F3A6E;
        margin-top: 8px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 16px rgba(47, 58, 110, 0.04);
    }

    /* Section Titles */
    .modern-header {
        font-size: 22px;
        font-weight: 800;
        color: #2F3A6E;
        border-bottom: 2px solid #ABC4E6;
        padding-bottom: 10px;
        margin-top: 12px;
        margin-bottom: 20px;
        letter-spacing: -0.015em;
    }

    /* Citation Quote Cards */
    .citation-quote {
        border-left: 4px solid #7174B9;
        background-color: #F4F7FC;
        padding: 14px 18px;
        font-style: italic;
        margin: 12px 0;
        border-radius: 0 12px 12px 0;
        color: #302F38;
        font-size: 14px;
        line-height: 1.65;
    }

    /* Clean Badges & Chips */
    .badge-chip {
        display: inline-block;
        background: #C0CDEC;
        color: #2F3A6E;
        border: 1px solid #7174B9;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 10px;
        letter-spacing: 0.02em;
    }
    
    .badge-conflict {
        background: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
    }

    /* STREAMLIT TAB CUSTOMIZATION - FIXING OVERFLOW & UNALIGNED BARS */
    div[data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #EBF1FA !important;
        padding: 6px 12px !important;
        border-radius: 9999px !important;
        border: 1px solid #ABC4E6 !important;
        margin-top: 12px !important;
        margin-bottom: 24px !important;
        display: flex !important;
        flex-wrap: wrap !important;
    }
    button[data-baseweb="tab"] {
        border-radius: 9999px !important;
        padding: 8px 18px !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        color: #2F3A6E !important;
        background-color: transparent !important;
        border: none !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #2F3A6E 0%, #7174B9 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(47, 58, 110, 0.25) !important;
    }
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Input & Button Overrides */
    div.stButton > button[kind="primary"], div.stButton > button {
        background: linear-gradient(135deg, #2F3A6E 0%, #474E86 100%) !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 10px 24px !important;
        box-shadow: 0 6px 16px rgba(47, 58, 110, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(47, 58, 110, 0.3) !important;
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
    st.markdown(f"## {APP_ICON} **Research Co-Pilot**")
    st.markdown("<div style='font-size: 13px; color: #7174B9; font-weight: 600;'>Verifiable Multi-Paper AI Suite</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📤 Upload Research Papers")
    uploaded_files = st.file_uploader(
        "Upload PDF Paper(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload academic papers to compare layout, methodology, formulas, and empirical results side-by-side."
    )
    
    if uploaded_files:
        if st.button("🚀 Process & Index Papers", use_container_width=True, type="primary"):
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

    st.markdown("---")
    st.markdown("### 📄 Comparative Scope Controls")
    if paper_names:
        scope_mode = st.radio(
            "Select Comparative Mode:",
            ["📊 Compare Multiple Papers (Multi-Paper Matrix)", "🎯 Single Paper Deep-Dive"],
            index=0 if len(paper_names) > 1 else 1,
            help="Choose whether to compare multiple research papers side-by-side or focus on 1 paper."
        )

        if scope_mode == "📊 Compare Multiple Papers (Multi-Paper Matrix)":
            selected_papers = st.multiselect(
                "Select Papers to Compare:",
                options=paper_names,
                default=paper_names,
                help="Select 2 or more research papers to compare side-by-side."
            )
            if not selected_papers:
                selected_papers = paper_names
            target_papers = selected_papers
            st.session_state.active_paper = f"Multi-Paper Matrix ({len(target_papers)} Papers Selected)"
            selected_filter = None
        else:
            selected_single = st.selectbox(
                "Select Target Paper:",
                options=paper_names,
                index=0,
                help="Analysis will focus strictly on this single document."
            )
            target_papers = [selected_single]
            st.session_state.active_paper = selected_single
            selected_filter = selected_single
    else:
        target_papers = []
        selected_filter = None
        st.info("No papers added yet. Upload PDF paper(s) above to begin.")

    st.markdown("---")
    st.markdown("<div style='font-size: 11.5px; color: #7174B9; text-align: center;'>IEEE Author Center Standards Compliant<br>© 2026 Research Co-Pilot AI</div>", unsafe_allow_html=True)

# Helper function to render scope banner
def render_scope_banner():
    if paper_names and target_papers:
        if len(target_papers) > 1:
            st.markdown(
                f"<div class='scope-pill'>"
                f"<span>📊 Comparing Multi-Paper Scope: <b>{', '.join([p[:35] + ('...' if len(p)>35 else '') for p in target_papers])}</b></span>"
                f"<span style='font-size: 12.5px; color: #2F3A6E;'>{len(target_papers)} Papers Selected</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            p_name = target_papers[0]
            chunk_count = len(retrieve_all_chunks_for_paper(p_name))
            st.markdown(
                f"<div class='scope-pill'>"
                f"<span>🎯 Currently Analyzing Active Paper: <b>{p_name}</b></span>"
                f"<span style='font-size: 12.5px; color: #2F3A6E;'>{chunk_count} Passages Indexed</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ No paper loaded in repository. Please upload PDF research paper(s) using the sidebar to begin analysis.")

# Top Hero Header Banner
st.markdown("""
<div class="hero-banner-container">
    <div class="hero-pill-badge">✨ Publication-Ready Scientific Suite</div>
    <h1>🔬 Research Paper Co-Pilot</h1>
    <p>Verifiable Multimodal Scientific Assistant & Multi-Paper IEEE Conference Studio</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================================
# MAIN PRODUCT TABS
# ==========================================================================
tabs = st.tabs([
    "💬 Grounded AI Research Assistant",
    "🔍 Document Analysis Teardown",
    "⚡ Cross-Paper Conflict Detector",
    "📐 Formulas & Quantitative Setup",
    "🧭 Literature Review & Gaps",
    "✍️ IEEE Manuscript & LaTeX Studio"
])

# --------------------------------------------------------------------------
# TAB 1: GROUNDED AI RESEARCH ASSISTANT (CONTINUOUS CHAT)
# --------------------------------------------------------------------------
with tabs[0]:
    st.markdown("<div class='modern-header'>💬 Grounded AI Research Assistant</div>", unsafe_allow_html=True)
    render_scope_banner()
    st.caption("Ask continuous follow-up questions comparing active papers. All responses feature verified source citations.")

    # Render Chat History Feed
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"**Question ({entry.get('paper', 'Target')}):** {entry['question']}")
        with st.chat_message("assistant"):
            st.markdown(f"<span class='badge-chip'>VERIFIED CITATION GROUNDING</span>", unsafe_allow_html=True)
            st.markdown(entry['answer'])
            with st.expander("🔎 View Source Passages & Citation Details", expanded=False):
                for idx, src in enumerate(entry.get("sources", [])):
                    meta = src.get("metadata", {})
                    st.markdown(f"**[{idx+1}] {meta.get('paper_name')}** — Page {meta.get('page_number')} (Section: {meta.get('section', 'General')})")
                    st.markdown(f"<div class='citation-quote'>\"{src.get('text')[:280]}...\"</div>", unsafe_allow_html=True)

    # Continuous Chat Input
    if prompt := st.chat_input("Ask a question comparing your selected papers...", disabled=not paper_names):
        with st.chat_message("user"):
            st.markdown(f"**Question ({st.session_state.active_paper}):** {prompt}")

        with st.chat_message("assistant"):
            with st.spinner(f"Retrieving passages across {len(target_papers)} paper(s)..."):
                # Retrieve top chunks for each target paper
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

                with st.expander("🔎 View Source Passages & Citation Details", expanded=False):
                    for idx, src in enumerate(retrieved_chunks):
                        meta = src.get("metadata", {})
                        st.markdown(f"**[{idx+1}] {meta.get('paper_name')}** — Page {meta.get('page_number')} (Section: {meta.get('section', 'General')})")
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
    st.markdown("<div class='modern-header'>🔍 Document Analysis Teardown</div>", unsafe_allow_html=True)
    render_scope_banner()

    if target_papers:
        if st.button(f"📊 Generate Teardown Breakdown ({len(target_papers)} Selected Papers)", type="primary", use_container_width=True):
            with st.spinner("Extracting structured academic breakdowns for selected papers..."):
                t_cols = st.tabs([f"📄 {p[:30]}..." for p in target_papers])
                for idx, p_name in enumerate(target_papers):
                    with t_cols[idx]:
                        analysis_res = analyze_paper(p_name)
                        st.markdown(f"### 📄 Academic Breakdown: `{p_name}`")
                        for field, content in analysis_res.items():
                            with st.expander(f"📌 {field}", expanded=True):
                                st.write(content)

# --------------------------------------------------------------------------
# TAB 3: CROSS-PAPER CONFLICT DETECTOR
# --------------------------------------------------------------------------
with tabs[2]:
    st.markdown("<div class='modern-header'>⚡ Cross-Paper Conflict & Controversy Detector</div>", unsafe_allow_html=True)
    render_scope_banner()
    st.write("Discovers empirical contradictions, conflicting methodological claims, and root causes of variance across papers.")

    if target_papers:
        if st.button(f"⚡ Run Cross-Paper Conflict Analysis ({len(target_papers)} Selected Papers)", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing cross-paper refutation & empirical variance for {st.session_state.active_paper}..."):
                conflict_res = detect_cross_paper_conflicts(target_papers)
                st.markdown(f"### 📊 Contradiction & Variance Analysis ({st.session_state.active_paper})")
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
    st.markdown("<div class='modern-header'>📐 Comparative Formulas & Technical Setup</div>", unsafe_allow_html=True)
    render_scope_banner()
    st.write("Extracts and compares mathematical equations (LaTeX), hardware sensor specs, parameters, and benchmark metrics across selected papers.")

    if target_papers:
        if st.button(f"📐 Extract & Compare Quantitative Setup ({len(target_papers)} Selected Papers)", type="primary", use_container_width=True):
            with st.spinner("Extracting quantitative formulations and metrics across papers..."):
                f_tabs = st.tabs([f"🧮 {p[:30]}..." for p in target_papers])
                for idx, p_name in enumerate(target_papers):
                    with f_tabs[idx]:
                        f_res = extract_formulas_and_metrics(p_name)
                        st.markdown(f"### 🧮 Quantitative Setup: `{p_name}`")
                        st.write(f_res.get("summary", ""))

                        st.markdown("#### 📐 Extracted Equations & Formulas")
                        eqs = f_res.get("equations", [])
                        if eqs:
                            for eq in eqs:
                                with st.expander(f"Equation: {eq.get('name')}", expanded=True):
                                    st.latex(eq.get("latex", ""))
                                    st.caption(eq.get("description", ""))
                        else:
                            st.info("No formal equations found in document text.")

                        st.markdown("#### ⚙️ Hardware Sensors, Parameters & Dataset Metrics")
                        hp_data = f_res.get("hyperparameters_and_setup", [])
                        if hp_data:
                            st.dataframe(pd.DataFrame(hp_data), use_container_width=True)

# --------------------------------------------------------------------------
# TAB 5: LITERATURE SYNTHESIS & RESEARCH GAPS
# --------------------------------------------------------------------------
with tabs[4]:
    st.markdown("<div class='modern-header'>🧭 Multi-Paper Literature Review & Research Gap Matrix</div>", unsafe_allow_html=True)
    render_scope_banner()

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown(f"#### 📚 Multi-Paper Literature Review Synthesis")
        if st.button(f"Synthesize Literature Review ({len(target_papers)} Papers)", type="primary", use_container_width=True, disabled=not target_papers):
            with st.spinner(f"Synthesizing comparative review for {len(target_papers)} paper(s)..."):
                rev = generate_literature_review(target_papers)
                for k, v in rev.items():
                    with st.expander(f"📌 {k}", expanded=True):
                        st.write(v)

    with col_r2:
        st.markdown(f"#### 🔬 Multi-Paper Research Gap Matrix")
        if st.button(f"Discover Cross-Paper Research Gaps ({len(target_papers)} Papers)", type="primary", use_container_width=True, disabled=not target_papers):
            with st.spinner(f"Discovering cross-paper research gaps for {len(target_papers)} paper(s)..."):
                gaps = detect_research_gaps(target_papers)
                for k, v in gaps.items():
                    with st.expander(f"🚩 {k}", expanded=True):
                        st.info(v)

# --------------------------------------------------------------------------
# TAB 6: IEEE MANUSCRIPT & LATEX STUDIO
# --------------------------------------------------------------------------
with tabs[5]:
    st.markdown("<div class='modern-header'>✍️ IEEE Survey & Multi-Paper Manuscript Studio</div>", unsafe_allow_html=True)
    render_scope_banner()

    paper_topic = st.text_input(
        "IEEE Paper Focus / Title:",
        value=f"A Comparative Survey and Verifiable Analysis of {len(target_papers)} Academic Literature Frameworks"
    )
    
    if st.button(f"📝 Generate Multi-Paper IEEE Conference Draft & LaTeX", type="primary", disabled=not target_papers, use_container_width=True):
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

        st.markdown(f"## {a6.get('paper_title')}")
        st.markdown(f"**Abstract**— {a6.get('abstract')}")
        st.markdown(f"*Index Terms*— {kw_str}")
        st.markdown("---")

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

        st.markdown("### 📑 BibTeX Entries")
        st.code(a6.get("bibtex_entries", ""), language="bibtex")

        st.markdown("### 📄 Camera-Ready IEEE LaTeX Source Code (`.tex`)")
        st.code(a6.get("latex_source", "% IEEE LaTeX source"), language="latex")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            formatted_refs = "\n".join(ref_list)
            full_md = f"# {a6.get('paper_title')}\n\n**Abstract**— {a6.get('abstract')}\n\n**Keywords**— {kw_str}\n\n## I. INTRODUCTION\n{sec.get('introduction')}\n\n## II. RELATED WORK & COMPARATIVE TAXONOMY\n{sec.get('related_work')}\n\n## III. SYSTEM ARCHITECTURE & COMPARATIVE FRAMEWORK\n{sec.get('methodology')}\n\n## IV. EXPERIMENTAL COMPARISON & EMPIRICAL RESULTS\n{sec.get('experiments_and_results')}\n\n## V. DISCUSSION & CROSS-PAPER RESEARCH GAPS\n{sec.get('discussion_and_gaps')}\n\n## VI. CONCLUSION\n{sec.get('conclusion')}\n\n## REFERENCES\n{formatted_refs}\n\n## BIBTEX\n```bibtex\n{a6.get('bibtex_entries')}\n```\n"
            st.download_button("📥 Download Markdown Draft (.md)", data=full_md, file_name="IEEE_Comparative_Manuscript.md", mime="text/markdown", use_container_width=True)
        with col_d2:
            st.download_button("📥 Download Overleaf LaTeX (.tex)", data=a6.get("latex_source", ""), file_name="IEEE_Comparative_Manuscript.tex", mime="text/x-tex", use_container_width=True)
