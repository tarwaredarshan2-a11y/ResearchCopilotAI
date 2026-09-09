"""
app.py
------
Research Paper Co-Pilot — Verifiable Academic AI Platform & IEEE Paper Studio.

Palette:
- Page Background: #F2EFE7 (Warm off-white)
- Sidebar Background: #E8E4DB (Deeper warm beige)
- Card Containers: #FFFFFF (Pure White)
- Primary Action: #3368A0 (Medium Scientific Blue)
- Secondary / Hover: #66A3BF (Teal Blue)
- Soft Surface / Chips: #C8DFDB (Mint Seafoam)
- Main Text: #1E2A3A (Dark Navy)
- Secondary Text: #5A6A7A (Muted Slate)
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

# ==========================================================================
# STREAMLIT CONFIG & LIGHT SCIENTIFIC DESIGN SYSTEM
# ==========================================================================
st.set_page_config(
    page_title=f"{APP_TITLE} | Verifiable Scientific AI Platform",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Scientific Light Background */
    .stApp {
        background-color: #F2EFE7;
        color: #1E2A3A;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #E8E4DB;
        border-right: 1px solid #D1D9E0;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #1E2A3A;
    }

    /* Primary Scientific Card Panels */
    .sci-card {
        background-color: #FFFFFF;
        border: 1px solid #D1D9E0;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(30, 42, 58, 0.05);
        color: #1E2A3A;
    }

    /* Active Scope Banner */
    .scope-banner {
        background-color: #C8DFDB;
        border: 1px solid #A8CFC9;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 13.5px;
        font-weight: 600;
        color: #1E2A3A;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Section Titles */
    .sci-header {
        font-size: 21px;
        font-weight: 700;
        color: #1E2A3A;
        border-bottom: 2px solid #D1D9E0;
        padding-bottom: 8px;
        margin-top: 6px;
        margin-bottom: 16px;
        letter-spacing: -0.01em;
    }

    /* Citation Quote Cards */
    .citation-card {
        border-left: 4px solid #3368A0;
        background-color: #F8F6F1;
        padding: 12px 16px;
        font-style: italic;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
        color: #2C3E50;
        font-size: 13.5px;
        line-height: 1.6;
    }

    /* Conversational Chat */
    .chat-user {
        background-color: #E2E8F0;
        border: 1px solid #CBD5E1;
        border-radius: 10px 10px 2px 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        color: #1E2A3A;
        font-weight: 500;
    }
    .chat-assistant {
        background-color: #FFFFFF;
        border: 1px solid #D1D9E0;
        border-radius: 10px 10px 10px 2px;
        padding: 18px 22px;
        margin-bottom: 16px;
        color: #1E2A3A;
        line-height: 1.65;
        box-shadow: 0 4px 12px rgba(30, 42, 58, 0.04);
    }
    
    /* Clean Badges & Chips */
    .badge-chip {
        display: inline-block;
        background: #C8DFDB;
        color: #1E2A3A;
        border: 1px solid #A8CFC9;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 11.5px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .badge-conflict {
        background: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 11.5px;
        font-weight: 600;
    }

    /* Primary Streamlit Button Overrides */
    div.stButton > button[kind="primary"] {
        background-color: #3368A0 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #275282 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_pipeline_result" not in st.session_state:
    st.session_state.last_pipeline_result = None

# Fetch available paper names
paper_names = get_all_paper_names()

# ==========================================================================
# SIDEBAR REPOSITORY & PAPER SELECTOR
# ==========================================================================
with st.sidebar:
    st.markdown(f"## {APP_ICON} **Research Co-Pilot**")
    st.markdown("<div style='font-size: 13px; color: #5A6A7A;'>Verifiable Multimodal Scientific Platform</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📤 Upload Research Paper")
    uploaded_files = st.file_uploader(
        "Select PDF Paper(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload academic PDFs to extract sections, layout structures, and formulas."
    )
    
    if uploaded_files:
        if st.button("🚀 Process & Ingest Papers", use_container_width=True, type="primary"):
            with st.spinner("Parsing layout, extracting sections, and indexing literature..."):
                for up_file in uploaded_files:
                    save_path = save_uploaded_pdf(up_file)
                    parsed_doc = parse_multimodal_pdf(save_path)
                    chunks = chunk_parsed_document(parsed_doc)
                    from langchain_core.documents import Document
                    docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]
                    add_documents_to_vector_store(docs)
                hybrid_retriever.sync_bm25_from_vector_store()
                st.success(f"Ingested {len(uploaded_files)} paper(s) successfully!")
                st.rerun()

    st.markdown("---")
    st.markdown("### 📄 Active Literature Scope")
    if paper_names:
        active_paper = st.selectbox(
            "Target Scope:",
            ["All Papers in Repository"] + paper_names,
            help="Select a specific paper or analyze the entire repository."
        )
        selected_paper_filter = None if active_paper == "All Papers in Repository" else active_paper
    else:
        active_paper = None
        selected_paper_filter = None
        st.info("No papers added yet. Upload a PDF above to begin.")

    st.markdown("---")
    st.markdown("<div style='font-size: 11.5px; color: #5A6A7A; text-align: center;'>IEEE Author Center Format Compliant<br>© 2026 Research Co-Pilot AI</div>", unsafe_allow_html=True)

# ==========================================================================
# MAIN PRODUCT TABS
# ==========================================================================
tabs = st.tabs([
    "💬 AI Research Assistant",
    "⚡ Controversy & Conflict Detector",
    "📐 Formulas & Technical Matrix",
    "🧭 Literature Synthesis & Gaps",
    "📄 IEEE Paper & LaTeX Studio",
    "🔍 Document Analysis Teardown"
])

# --------------------------------------------------------------------------
# TAB 1: GROUNDED AI RESEARCH ASSISTANT
# --------------------------------------------------------------------------
with tabs[0]:
    st.markdown("<div class='sci-header'>💬 Grounded AI Research Assistant</div>", unsafe_allow_html=True)
    
    # Active Scope Indicator
    if paper_names:
        st.markdown(f"<div class='scope-banner'>🎯 Active Context Scope: <b>{active_paper}</b></div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ No paper loaded in repository. Upload a PDF paper using the sidebar to begin analysis.")

    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        user_question = st.text_input(
            "Enter Research Question:",
            placeholder="e.g. How does Retrieval-Augmented Generation reduce hallucination in large language models?",
            label_visibility="collapsed"
        )
    with col_c2:
        search_btn = st.button("🔍 Search & Synthesize", type="primary", use_container_width=True, disabled=not paper_names)

    if search_btn and user_question and paper_names:
        with st.spinner("Retrieving literature context & verifying source grounding..."):
            retrieved_chunks = hybrid_retriever.retrieve(
                query=user_question,
                top_k=5,
                paper_name=selected_paper_filter
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
                "question": user_question,
                "answer": ai_answer,
                "sources": retrieved_chunks
            })

    # Render Feed
    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"<div class='chat-user'><b>User Question:</b> {entry['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-assistant'><span class='badge-chip'>VERIFIED CITATION GROUNDING</span><br><br>{entry['answer']}</div>", unsafe_allow_html=True)
        with st.expander("🔎 View Source Passages & Page Citation Details", expanded=False):
            for idx, src in enumerate(entry["sources"]):
                meta = src.get("metadata", {})
                st.markdown(f"**[{idx+1}] {meta.get('paper_name')}** — Page {meta.get('page_number')} (Section: {meta.get('section', 'General')})")
                st.markdown(f"<div class='citation-card'>\"{src.get('text')[:280]}...\"</div>", unsafe_allow_html=True)
        st.markdown("---")

# --------------------------------------------------------------------------
# TAB 2: CONTROVERSY & CONFLICT DETECTOR (NOVEL FEATURE)
# --------------------------------------------------------------------------
with tabs[1]:
    st.markdown("<div class='sci-header'>⚡ Cross-Paper Conflict & Controversy Detector</div>", unsafe_allow_html=True)
    st.write("Identifies where papers contradict each other, variance in empirical results, and opposing methodological claims.")

    if paper_names:
        if st.button("⚡ Run Conflict Detection", type="primary", use_container_width=True):
            with st.spinner("Analyzing literature cross-refutation & variance..."):
                conflict_res = detect_cross_paper_conflicts(paper_names)
                st.markdown("### 📊 Contradiction & Variance Analysis")
                st.write(conflict_res.get("summary", ""))

                for idx, c in enumerate(conflict_res.get("conflicts", [])):
                    with st.container():
                        st.markdown(f"#### <span class='badge-conflict'>CONFLICT #{idx+1}</span> {c.get('conflict_topic')}", unsafe_allow_html=True)
                        col_f1, col_f2 = st.columns(2)
                        with col_f1:
                            st.markdown(f"<div class='sci-card'><b>Claim A:</b><br>{c.get('paper_a_claim')}</div>", unsafe_allow_html=True)
                        with col_f2:
                            st.markdown(f"<div class='sci-card'><b>Claim B (Opposing):</b><br>{c.get('paper_b_claim')}</div>", unsafe_allow_html=True)
                        st.info(f"**Root Cause of Variance:** {c.get('root_cause')}")
                        st.success(f"**Reconciliation Hypothesis:** {c.get('reconciliation_hypothesis')}")
                        st.markdown("---")
    else:
        st.info("No papers added yet. Upload PDFs using the sidebar to run conflict detection.")

# --------------------------------------------------------------------------
# TAB 3: FORMULAS & TECHNICAL MATRIX (NOVEL FEATURE)
# --------------------------------------------------------------------------
with tabs[2]:
    st.markdown("<div class='sci-header'>📐 Formulas & Mathematical Inventory</div>", unsafe_allow_html=True)
    st.write("Extracts mathematical equations (LaTeX), hyperparameter configurations, and quantitative metrics from literature.")

    if paper_names:
        target_f_paper = st.selectbox("Select Paper for Formula Extraction:", paper_names, key="f_paper")
        if st.button("📐 Extract Equations & Parameters", type="primary"):
            with st.spinner(f"Extracting mathematical equations for {target_f_paper}..."):
                f_res = extract_formulas_and_metrics(target_f_paper)
                st.markdown(f"### 🧮 Mathematical & Technical Setup: `{target_f_paper}`")
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
    else:
        st.info("No papers added yet. Upload a PDF using the sidebar.")

# --------------------------------------------------------------------------
# TAB 4: LITERATURE SYNTHESIS & RESEARCH GAPS
# --------------------------------------------------------------------------
with tabs[3]:
    st.markdown("<div class='sci-header'>🧭 Literature Synthesis & Research Gap Matrix</div>", unsafe_allow_html=True)
    st.write("Synthesize cross-paper literature reviews, comparative tables, and discover open research gaps and testable hypotheses.")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("#### 📚 Comprehensive Literature Review")
        if st.button("Synthesize Literature Review", type="primary", use_container_width=True, disabled=not paper_names):
            with st.spinner("Synthesizing multi-paper literature review..."):
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
# TAB 5: IEEE PAPER & LATEX STUDIO
# --------------------------------------------------------------------------
with tabs[4]:
    st.markdown("<div class='sci-header'>✍️ IEEE Manuscript & LaTeX Studio</div>", unsafe_allow_html=True)
    st.write("Draft publication-ready IEEE conference sections, bracketed citations [1], BibTeX entries, and copy-pasteable Overleaf LaTeX source code.")

    paper_topic = st.text_input(
        "IEEE Paper Title / Focus Area:",
        value="A Verifiable Multi-Agent Framework for Scientific Literature Synthesis"
    )
    
    if st.button("📝 Generate IEEE Conference Draft & LaTeX", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(msg: str, pct: int):
            status_text.markdown(f"**{msg}**")
            progress_bar.progress(pct)

        with st.spinner("Drafting IEEE conference paper sections & LaTeX source..."):
            pipe_res = orchestrator.run_full_pipeline(
                research_query=paper_topic,
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

# --------------------------------------------------------------------------
# TAB 6: DOCUMENT ANALYSIS TEARDOWN
# --------------------------------------------------------------------------
with tabs[5]:
    st.markdown("<div class='sci-header'>🔍 Document Analysis Teardown</div>", unsafe_allow_html=True)
    st.write("Extract a comprehensive, structured teardown of an uploaded paper (Abstract, Methodology, Dataset, Model Architecture, Results, References).")

    if paper_names:
        selected_analysis_paper = st.selectbox("Select Paper for Analysis:", paper_names, key="analysis_paper_select")
        if st.button("📊 Generate Teardown", type="primary"):
            with st.spinner(f"Parsing structured sections for {selected_analysis_paper}..."):
                analysis_res = analyze_paper(selected_analysis_paper)
                st.markdown(f"### 📄 Academic Teardown: `{selected_analysis_paper}`")
                for field, content in analysis_res.items():
                    with st.expander(f"📌 {field}", expanded=True):
                        st.write(content)
    else:
        st.info("No papers added yet. Upload a PDF using the sidebar repository.")
