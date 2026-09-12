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
    page_title=f"{APP_TITLE} | Verifiable Scientific Platform",
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
    
    /* Main Page Canvas Background - Soft Warm Crisp Light */
    .stApp {
        background: linear-gradient(180deg, #EBF1FA 0%, #F5F7FC 400px, #F5F7FC 100%);
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

    /* Organic Curved Hero Header Banner (Inspired by Reference Images 1 & 2) */
    .hero-banner-container {
        position: relative;
        background: linear-gradient(135deg, #2F3A6E 0%, #474E86 45%, #7174B9 100%);
        border-radius: 24px;
        padding: 36px 42px 54px 42px;
        color: #FFFFFF;
        margin-bottom: -24px;
        box-shadow: 0 16px 36px rgba(47, 58, 110, 0.18);
        overflow: hidden;
    }
    .hero-banner-container::after {
        content: "";
        position: absolute;
        bottom: -1px;
        left: 0;
        right: 0;
        height: 36px;
        background: #F5F7FC;
        clip-path: ellipse(55% 100% at 50% 100%);
    }
    .hero-banner-container h1 {
        color: #FFFFFF !important;
        font-size: 32px;
        font-weight: 800;
        margin: 0 0 8px 0;
        letter-spacing: -0.025em;
        text-shadow: 0 2px 10px rgba(0,0,0,0.12);
    }
    .hero-banner-container p {
        color: #C0CDEC !important;
        font-size: 15px;
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
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 12.5px;
        font-weight: 700;
        margin-bottom: 14px;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
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
        margin-top: 10px;
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

    /* Conversational Chat */
    .chat-user {
        background: linear-gradient(135deg, #E6EDF8 0%, #DCE5F5 100%);
        border: 1px solid #ABC4E6;
        border-radius: 18px 18px 4px 18px;
        padding: 16px 20px;
        margin-bottom: 14px;
        color: #2F3A6E;
        font-weight: 600;
    }
    .chat-assistant {
        background-color: #FFFFFF;
        border: 1px solid #ABC4E6;
        border-radius: 18px 18px 18px 4px;
        padding: 22px 26px;
        margin-bottom: 20px;
        color: #302F38;
        line-height: 1.7;
        box-shadow: 0 8px 24px rgba(47, 58, 110, 0.06);
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

    /* Streamlit Tab Customization - Modern Pill Navigation */
    button[data-baseweb="tab"] {
        border-radius: 9999px !important;
        padding: 10px 22px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        color: #7174B9 !important;
        background-color: transparent !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #2F3A6E 0%, #7174B9 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 18px rgba(47, 58, 110, 0.25) !important;
    }
    div[data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Streamlit Input & Selectbox Styling */
    div[data-baseweb="input"] input {
        border-radius: 12px !important;
        color: #302F38 !important;
    }

    /* Streamlit Primary Buttons Overrides (#2F3A6E) */
    div.stButton > button[kind="primary"], div.stButton > button {
        background: linear-gradient(135deg, #2F3A6E 0%, #474E86 100%) !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 12px 26px !important;
        box-shadow: 0 6px 16px rgba(47, 58, 110, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover, div.stButton > button:hover {
        background: linear-gradient(135deg, #212952 0%, #393F6E 100%) !important;
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
if "active_paper" not in st.session_state:
    st.session_state.active_paper = None

# Fetch paper list from database/uploads
paper_names = get_all_paper_names()

# Ensure active_paper session state is valid
if paper_names and (st.session_state.active_paper not in paper_names and st.session_state.active_paper != "All Papers in Repository"):
    st.session_state.active_paper = paper_names[0]

# ==========================================================================
# SIDEBAR REPOSITORY & PAPER SELECTION
# ==========================================================================
with st.sidebar:
    st.markdown(f"## {APP_ICON} **Research Co-Pilot**")
    st.markdown("<div style='font-size: 13px; color: #7174B9; font-weight: 600;'>Verifiable Scientific AI Platform</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📤 Upload Research Paper")
    uploaded_files = st.file_uploader(
        "Upload PDF Paper(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload academic papers to analyze text, layout, formulas, and references."
    )
    
    if uploaded_files:
        if st.button("🚀 Process & Select Paper", use_container_width=True, type="primary"):
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
                if last_name:
                    st.session_state.active_paper = last_name
                st.success(f"Indexed & selected '{st.session_state.active_paper}'!")
                st.rerun()

    st.markdown("---")
    st.markdown("### 📄 Active Document Selector")
    if paper_names:
        options = ["All Papers in Repository"] + paper_names
        current_idx = options.index(st.session_state.active_paper) if st.session_state.active_paper in options else 0
        selected_paper_option = st.selectbox(
            "Select Active Paper:",
            options,
            index=current_idx,
            help="All analysis, Q&A, and drafting will be strictly scoped to your selected paper."
        )
        st.session_state.active_paper = selected_paper_option
        selected_filter = None if selected_paper_option == "All Papers in Repository" else selected_paper_option
    else:
        selected_filter = None
        st.info("No papers added yet. Upload a PDF above to begin.")

    st.markdown("---")
    st.markdown("<div style='font-size: 11.5px; color: #7174B9; text-align: center;'>IEEE Author Center Standards Compliant<br>© 2026 Research Co-Pilot AI</div>", unsafe_allow_html=True)

# Helper function to render active paper banner
def render_scope_banner():
    if paper_names and st.session_state.active_paper:
        target_display = st.session_state.active_paper
        if target_display != "All Papers in Repository":
            chunk_count = len(retrieve_all_chunks_for_paper(target_display))
            st.markdown(
                f"<div class='scope-pill'>"
                f"<span>🎯 Currently Analyzing Active Paper: <b>{target_display}</b></span>"
                f"<span style='font-size: 12.5px; color: #2F3A6E;'>{chunk_count} Passages Indexed</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='scope-pill'>"
                f"<span>🎯 Currently Analyzing Scope: <b>All {len(paper_names)} Papers in Repository</b></span>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ No paper loaded in repository. Please upload a PDF research paper using the sidebar to begin analysis.")

# Top Hero Header Banner (Inspired by Reference Designs 1 & 2)
st.markdown("""
<div class="hero-banner-container">
    <div class="hero-pill-badge">✨ Publication-Ready Scientific Suite</div>
    <h1>🔬 Research Paper Co-Pilot</h1>
    <p>Verifiable Multimodal Scientific Assistant & IEEE Conference Manuscript Studio</p>
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
# TAB 1: GROUNDED AI RESEARCH ASSISTANT
# --------------------------------------------------------------------------
with tabs[0]:
    st.markdown("<div class='modern-header'>💬 Grounded AI Research Assistant</div>", unsafe_allow_html=True)
    render_scope_banner()

    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        user_question = st.text_input(
            "Enter Question for Active Document:",
            placeholder="e.g. What methodology, datasets, and key findings are presented in this paper?",
            label_visibility="collapsed"
        )
    with col_c2:
        search_btn = st.button("🔍 Search & Synthesize", type="primary", use_container_width=True, disabled=not paper_names)

    if search_btn and user_question and paper_names:
        with st.spinner(f"Retrieving passages from {st.session_state.active_paper}..."):
            retrieved_chunks = hybrid_retriever.retrieve(
                query=user_question,
                top_k=5,
                paper_name=selected_filter
            )
            
            context_str = "\n\n".join([
                f"[Source: {c.get('metadata', {}).get('paper_name')} | Page: {c.get('metadata', {}).get('page_number')}]\n{c.get('text')}"
                for c in retrieved_chunks
            ])

            chat_prompt = f"""You are an expert academic research assistant. Answer the user's question using ONLY the provided literature context.
Include bracketed citations [PaperName, p.X] for every key fact.

Literature Context:
{context_str}

User Question: {user_question}
"""
            ai_answer = generate_response(chat_prompt)
            st.session_state.chat_history.append({
                "paper": st.session_state.active_paper,
                "question": user_question,
                "answer": ai_answer,
                "sources": retrieved_chunks
            })

    # Render Feed
    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"<div class='chat-user'><b>Question ({entry.get('paper', 'Target')}):</b> {entry['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-assistant'><span class='badge-chip'>VERIFIED CITATION GROUNDING</span><br><br>{entry['answer']}</div>", unsafe_allow_html=True)
        with st.expander("🔎 View Source Passages & Page Citation Details", expanded=False):
            for idx, src in enumerate(entry["sources"]):
                meta = src.get("metadata", {})
                st.markdown(f"**[{idx+1}] {meta.get('paper_name')}** — Page {meta.get('page_number')} (Section: {meta.get('section', 'General')})")
                st.markdown(f"<div class='citation-quote'>\"{src.get('text')[:280]}...\"</div>", unsafe_allow_html=True)
        st.markdown("---")

# --------------------------------------------------------------------------
# TAB 2: DOCUMENT ANALYSIS TEARDOWN
# --------------------------------------------------------------------------
with tabs[1]:
    st.markdown("<div class='modern-header'>🔍 Document Analysis Teardown</div>", unsafe_allow_html=True)
    render_scope_banner()

    if paper_names and selected_filter:
        if st.button(f"📊 Generate Teardown for '{selected_filter}'", type="primary"):
            with st.spinner(f"Extracting structured sections for {selected_filter}..."):
                analysis_res = analyze_paper(selected_filter)
                st.markdown(f"### 📄 Academic Breakdown: `{selected_filter}`")
                for field, content in analysis_res.items():
                    with st.expander(f"📌 {field}", expanded=True):
                        st.write(content)
    elif paper_names and not selected_filter:
        st.info("Select a specific paper from the sidebar dropdown to run a single-document analysis teardown.")

# --------------------------------------------------------------------------
# TAB 3: CROSS-PAPER CONFLICT DETECTOR
# --------------------------------------------------------------------------
with tabs[2]:
    st.markdown("<div class='modern-header'>⚡ Cross-Paper Conflict & Controversy Detector</div>", unsafe_allow_html=True)
    render_scope_banner()
    st.write("Identifies where papers contradict each other, variance in empirical metrics, and opposing methodological assumptions.")

    if paper_names:
        if st.button("⚡ Run Cross-Paper Conflict Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing cross-paper literature refutation & empirical variance..."):
                conflict_res = detect_cross_paper_conflicts(paper_names)
                st.markdown("### 📊 Contradiction & Variance Analysis")
                st.write(conflict_res.get("summary", ""))

                for idx, c in enumerate(conflict_res.get("conflicts", [])):
                    with st.container():
                        st.markdown(f"#### <span class='badge-conflict'>CONFLICT #{idx+1}</span> {c.get('conflict_topic')}", unsafe_allow_html=True)
                        col_f1, col_f2 = st.columns(2)
                        with col_f1:
                            st.markdown(f"<div class='modern-card'><b>Claim A:</b><br>{c.get('paper_a_claim')}</div>", unsafe_allow_html=True)
                        with col_f2:
                            st.markdown(f"<div class='modern-card'><b>Claim B (Opposing):</b><br>{c.get('paper_b_claim')}</div>", unsafe_allow_html=True)
                        st.info(f"**Root Cause of Variance:** {c.get('root_cause')}")
                        st.success(f"**Reconciliation Hypothesis:** {c.get('reconciliation_hypothesis')}")
                        st.markdown("---")

# --------------------------------------------------------------------------
# TAB 4: FORMULAS & QUANTITATIVE SETUP
# --------------------------------------------------------------------------
with tabs[3]:
    st.markdown("<div class='modern-header'>📐 Formulas & Technical Setup</div>", unsafe_allow_html=True)
    render_scope_banner()
    st.write("Extracts mathematical equations (LaTeX), hyperparameter setups, and numerical metrics from the active paper.")

    if paper_names and selected_filter:
        if st.button(f"📐 Extract Formulas for '{selected_filter}'", type="primary"):
            with st.spinner(f"Extracting mathematical equations for {selected_filter}..."):
                f_res = extract_formulas_and_metrics(selected_filter)
                st.markdown(f"### 🧮 Mathematical & Technical Setup: `{selected_filter}`")
                st.write(f_res.get("summary", ""))

                st.markdown("#### 📐 Extracted Equations & Formulas")
                for eq in f_res.get("equations", []):
                    with st.expander(f"Equation: {eq.get('name')}", expanded=True):
                        st.latex(eq.get("latex", ""))
                        st.caption(eq.get("description", ""))

                st.markdown("#### ⚙️ Hyperparameters & Dataset Metrics")
                hp_data = f_res.get("hyperparameters_and_setup", [])
                if hp_data:
                    st.dataframe(pd.DataFrame(hp_data), use_container_width=True)
    elif paper_names and not selected_filter:
        st.info("Select a specific paper from the sidebar dropdown to extract formulas and technical metrics.")

# --------------------------------------------------------------------------
# TAB 5: LITERATURE SYNTHESIS & RESEARCH GAPS
# --------------------------------------------------------------------------
with tabs[4]:
    st.markdown("<div class='modern-header'>🧭 Literature Review & Research Gap Matrix</div>", unsafe_allow_html=True)
    render_scope_banner()

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("#### 📚 Literature Review Synthesis")
        if st.button("Synthesize Literature Review", type="primary", use_container_width=True, disabled=not paper_names):
            with st.spinner("Synthesizing literature review..."):
                rev = generate_literature_review(paper_names)
                for k, v in rev.items():
                    with st.expander(f"📌 {k}", expanded=True):
                        st.write(v)

    with col_r2:
        st.markdown("#### 🔬 Research Gap Matrix")
        if st.button("Discover Research Gaps", type="primary", use_container_width=True, disabled=not paper_names):
            with st.spinner("Analyzing limitations and open trajectories..."):
                gaps = detect_research_gaps(paper_names)
                for k, v in gaps.items():
                    with st.expander(f"🚩 {k}", expanded=True):
                        st.info(v)

# --------------------------------------------------------------------------
# TAB 6: IEEE MANUSCRIPT & LATEX STUDIO
# --------------------------------------------------------------------------
with tabs[5]:
    st.markdown("<div class='modern-header'>✍️ IEEE Manuscript & LaTeX Studio</div>", unsafe_allow_html=True)
    render_scope_banner()

    paper_topic = st.text_input(
        "IEEE Paper Focus / Title:",
        value=f"A Verifiable Multi-Agent Framework for Scientific Synthesis of {selected_filter if selected_filter else 'Academic Literature'}"
    )
    
    if st.button("📝 Generate IEEE Conference Draft & LaTeX", type="primary", disabled=not paper_names):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(msg: str, pct: int):
            status_text.markdown(f"**{msg}**")
            progress_bar.progress(pct)

        with st.spinner("Orchestrating 6-agent pipeline for IEEE conference paper synthesis..."):
            pipe_res = orchestrator.run_full_pipeline(
                research_query=paper_topic,
                selected_paper=selected_filter,
                progress_callback=update_progress
            )
            st.session_state.last_pipeline_result = pipe_res
            st.success("IEEE Conference Draft & LaTeX generated successfully!")

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
            ("related_work", "II. RELATED WORK"),
            ("methodology", "III. METHODOLOGY & SYSTEM ARCHITECTURE"),
            ("experiments_and_results", "IV. EXPERIMENTAL RESULTS & VERIFICATION"),
            ("discussion_and_gaps", "V. DISCUSSION & RESEARCH GAPS"),
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
            full_md = f"# {a6.get('paper_title')}\n\n**Abstract**— {a6.get('abstract')}\n\n**Keywords**— {kw_str}\n\n## I. INTRODUCTION\n{sec.get('introduction')}\n\n## II. RELATED WORK\n{sec.get('related_work')}\n\n## III. METHODOLOGY & SYSTEM ARCHITECTURE\n{sec.get('methodology')}\n\n## IV. EXPERIMENTAL RESULTS & VERIFICATION\n{sec.get('experiments_and_results')}\n\n## V. DISCUSSION & RESEARCH GAPS\n{sec.get('discussion_and_gaps')}\n\n## VI. CONCLUSION\n{sec.get('conclusion')}\n\n## REFERENCES\n{formatted_refs}\n\n## BIBTEX\n```bibtex\n{a6.get('bibtex_entries')}\n```\n"
            st.download_button("📥 Download Markdown Draft (.md)", data=full_md, file_name="IEEE_Manuscript_Draft.md", mime="text/markdown", use_container_width=True)
        with col_d2:
            st.download_button("📥 Download Overleaf LaTeX (.tex)", data=a6.get("latex_source", ""), file_name="IEEE_Manuscript.tex", mime="text/x-tex", use_container_width=True)
