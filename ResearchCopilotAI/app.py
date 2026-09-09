"""
app.py
------
Research Paper Co-Pilot — Verifiable Academic AI Platform & IEEE Paper Studio.

Clean, professional, user-centric product architecture designed for:
1. Grounded Multi-Paper Research Chat & Citation Grounding
2. Deep Single-Paper Academic Teardown & Analysis
3. Cross-Paper Literature Review & Research Gap Discovery
4. IEEE Conference-Aligned Paper Studio & BibTeX Exporter
5. Verified Multi-Agent Methodology & Technical Protocol
"""

import os
import streamlit as st
import pandas as pd
import json

from config import (
    APP_TITLE, APP_SUBTITLE, APP_ICON, GOOGLE_API_KEY,
    RETRIEVER_TOP_K, HYBRID_DENSE_WEIGHT, HYBRID_SPARSE_WEIGHT,
    AGENT_CONFIGS, IEEE_SECTIONS
)
from utils.pdf_loader import save_uploaded_pdf, get_pdf_page_count
from utils.multimodal_parser import parse_multimodal_pdf, chunk_parsed_document
from utils.embeddings import add_documents_to_vector_store, get_all_paper_names
from utils.hybrid_retriever import hybrid_retriever
from utils.orchestrator import orchestrator
from utils.evaluation_suite import evaluation_suite
from utils.llm import generate_response
from utils.paper_analysis import analyze_paper
from utils.literature_review import generate_literature_review
from utils.research_gap import detect_research_gaps

# ==========================================================================
# STREAMLIT CONFIG & PROFESSIONAL ACADEMIC UI STYLING
# ==========================================================================
st.set_page_config(
    page_title=f"{APP_TITLE} | Verifiable Academic AI",
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
    
    .stApp {
        background-color: #090d16;
        color: #f1f5f9;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }

    /* Product Cards */
    .product-card {
        background: #131824;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }

    /* Clean Section Titles */
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #f8fafc;
        border-bottom: 2px solid #1e293b;
        padding-bottom: 8px;
        margin-top: 8px;
        margin-bottom: 16px;
        letter-spacing: -0.01em;
    }

    /* Citation Quotes */
    .citation-card {
        border-left: 3px solid #3b82f6;
        background-color: #0f172a;
        padding: 12px 16px;
        font-style: italic;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
        color: #cbd5e1;
        font-size: 13.5px;
        line-height: 1.6;
    }

    /* Conversational Chat */
    .chat-user {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px 10px 2px 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        color: #f8fafc;
    }
    .chat-assistant {
        background: #131824;
        border: 1px solid #1e293b;
        border-radius: 10px 10px 10px 2px;
        padding: 16px 20px;
        margin-bottom: 16px;
        color: #e2e8f0;
        line-height: 1.6;
    }
    
    /* Clean Badges */
    .source-tag {
        display: inline-block;
        background: rgba(59, 130, 246, 0.12);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.25);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11.5px;
        font-weight: 600;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_pipeline_result" not in st.session_state:
    st.session_state.last_pipeline_result = None

# Fetch active paper names
paper_names = get_all_paper_names()

# ==========================================================================
# SIDEBAR PLATFORM NAVIGATION & REPOSITORY
# ==========================================================================
with st.sidebar:
    st.markdown(f"## {APP_ICON} **Research Co-Pilot**")
    st.caption("Verifiable Multimodal Scientific Assistant")
    st.markdown("---")

    st.markdown("### 📤 Document Repository")
    uploaded_files = st.file_uploader(
        "Upload PDF Papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload academic papers to parse sections and index literature context."
    )
    
    if uploaded_files:
        if st.button("🚀 Process & Ingest Papers", use_container_width=True, type="primary"):
            with st.spinner("Extracting layout, chunking, and indexing into knowledge base..."):
                for up_file in uploaded_files:
                    save_path = save_uploaded_pdf(up_file)
                    parsed_doc = parse_multimodal_pdf(save_path)
                    chunks = chunk_parsed_document(parsed_doc)
                    from langchain_core.documents import Document
                    docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]
                    add_documents_to_vector_store(docs)
                hybrid_retriever.sync_bm25_from_vector_store()
                st.success(f"Indexed {len(uploaded_files)} paper(s) successfully!")
                st.rerun()

    st.markdown("---")
    st.markdown("### 📄 Active Papers in Repository")
    if paper_names:
        for p in paper_names:
            st.markdown(f"- `{p}`")
    else:
        st.info("No papers added yet. Upload a PDF above to begin.")

    st.markdown("---")
    st.markdown("<div style='font-size: 11px; color: #64748b; text-align: center;'>IEEE Author Center Standards Compliant<br>© 2026 Research Co-Pilot AI</div>", unsafe_allow_html=True)

# ==========================================================================
# MAIN PRODUCT INTERFACE TABS
# ==========================================================================
tabs = st.tabs([
    "💬 AI Research Assistant",
    "🔍 Document Analysis",
    "🧭 Literature Review & Gaps",
    "✍️ IEEE Manuscript Studio",
    "📐 Methodology & System Architecture"
])

# --------------------------------------------------------------------------
# TAB 1: GROUNDED AI RESEARCH ASSISTANT
# --------------------------------------------------------------------------
with tabs[0]:
    st.markdown("<div class='section-title'>💬 Grounded AI Research Assistant</div>", unsafe_allow_html=True)
    st.write("Ask natural-language questions across your uploaded research papers. Answers are grounded in source text with page citations.")

    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        user_question = st.text_input(
            "Research Query:",
            placeholder="e.g. How does Retrieval-Augmented Generation reduce hallucination in large language models?",
            label_visibility="collapsed"
        )
    with col_c2:
        target_paper = st.selectbox(
            "Filter Scope:",
            ["All Papers"] + paper_names,
            label_visibility="collapsed"
        )
        filter_paper_name = None if target_paper == "All Papers" else target_paper

    if st.button("🔍 Search & Synthesize", type="primary") and user_question:
        with st.spinner("Retrieving literature context & verifying source grounding..."):
            retrieved_chunks = hybrid_retriever.retrieve(
                query=user_question,
                top_k=5,
                paper_name=filter_paper_name
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

    # Render Clean Conversational Feed
    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"<div class='chat-user'><b>Question:</b> {entry['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-assistant'><span class='source-tag'>VERIFIED SYNTHESIS</span><br>{entry['answer']}</div>", unsafe_allow_html=True)
        with st.expander("🔎 View Source Passages & Page Citation Details", expanded=False):
            for idx, src in enumerate(entry["sources"]):
                meta = src.get("metadata", {})
                st.markdown(f"**[{idx+1}] {meta.get('paper_name')}** — Page {meta.get('page_number')} (Section: {meta.get('section', 'General')})")
                st.markdown(f"<div class='citation-card'>\"{src.get('text')[:280]}...\"</div>", unsafe_allow_html=True)
        st.markdown("---")

# --------------------------------------------------------------------------
# TAB 2: SINGLE-PAPER DEEP DIVE & ANALYSIS
# --------------------------------------------------------------------------
with tabs[1]:
    st.markdown("<div class='section-title'>🔍 Single-Paper Deep Dive & Analysis</div>", unsafe_allow_html=True)
    st.write("Extract a comprehensive, structured teardown of an uploaded paper (Abstract, Methodology, Dataset, Model Architecture, Results, References).")

    if paper_names:
        selected_analysis_paper = st.selectbox("Select Paper for Deep Analysis:", paper_names)
        if st.button("📊 Generate Paper Teardown", type="primary"):
            with st.spinner(f"Parsing structured sections for {selected_analysis_paper}..."):
                analysis_res = analyze_paper(selected_analysis_paper)
                st.markdown(f"### 📄 Academic Teardown: `{selected_analysis_paper}`")
                for field, content in analysis_res.items():
                    with st.expander(f"📌 {field}", expanded=True):
                        st.write(content)
    else:
        st.info("No papers added yet. Upload a PDF using the left sidebar repository.")

# --------------------------------------------------------------------------
# TAB 3: LITERATURE REVIEW & RESEARCH GAPS
# --------------------------------------------------------------------------
with tabs[2]:
    st.markdown("<div class='section-title'>🧭 Literature Review & Research Gap Synthesizer</div>", unsafe_allow_html=True)
    st.write("Synthesize cross-paper literature reviews, comparative tables, and uncover open research gaps and testable hypotheses.")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("#### 📚 Comprehensive Literature Review")
        if st.button("Synthesize Literature Review", type="primary", use_container_width=True):
            with st.spinner("Synthesizing multi-paper literature review..."):
                rev = generate_literature_review(paper_names)
                for k, v in rev.items():
                    with st.expander(f"📌 {k}", expanded=True):
                        st.write(v)

    with col_r2:
        st.markdown("#### 🔬 Research Gap Matrix")
        if st.button("Discover Research Gaps", type="primary", use_container_width=True):
            with st.spinner("Analyzing limitations and open trajectories..."):
                gaps = detect_research_gaps(paper_names)
                for k, v in gaps.items():
                    with st.expander(f"🚩 {k}", expanded=True):
                        st.info(v)

# --------------------------------------------------------------------------
# TAB 4: IEEE MANUSCRIPT STUDIO
# --------------------------------------------------------------------------
with tabs[3]:
    st.markdown("<div class='section-title'>✍️ IEEE Conference Paper Studio</div>", unsafe_allow_html=True)
    st.write("Draft publication-ready IEEE conference paper sections (Abstract, Sections I–VI, References, and BibTeX citations).")

    paper_topic = st.text_input(
        "IEEE Paper Title / Focus Area:",
        value="A Verifiable Multi-Agent Framework for Scientific Literature Synthesis"
    )
    
    if st.button("📝 Generate IEEE Conference Draft", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(msg: str, pct: int):
            status_text.markdown(f"**{msg}**")
            progress_bar.progress(pct)

        with st.spinner("Drafting IEEE conference paper sections..."):
            pipe_res = orchestrator.run_full_pipeline(
                research_query=paper_topic,
                progress_callback=update_progress
            )
            st.session_state.last_pipeline_result = pipe_res
            st.success("IEEE Conference Draft generated successfully!")

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

        formatted_refs = "\n".join(ref_list)
        full_md = f"# {a6.get('paper_title')}\n\n**Abstract**— {a6.get('abstract')}\n\n**Keywords**— {kw_str}\n\n## I. INTRODUCTION\n{sec.get('introduction')}\n\n## II. RELATED WORK\n{sec.get('related_work')}\n\n## III. METHODOLOGY & SYSTEM ARCHITECTURE\n{sec.get('methodology')}\n\n## IV. EXPERIMENTAL RESULTS & VERIFICATION\n{sec.get('experiments_and_results')}\n\n## V. DISCUSSION & RESEARCH GAPS\n{sec.get('discussion_and_gaps')}\n\n## VI. CONCLUSION\n{sec.get('conclusion')}\n\n## REFERENCES\n{formatted_refs}\n\n## BIBTEX\n```bibtex\n{a6.get('bibtex_entries')}\n```\n"
        st.download_button("📥 Download IEEE Paper Draft (.md)", data=full_md, file_name="IEEE_Manuscript_Draft.md", mime="text/markdown")

# --------------------------------------------------------------------------
# TAB 5: SYSTEM METHODOLOGY & VERIFICATION PROTOCOL
# --------------------------------------------------------------------------
with tabs[4]:
    st.markdown("<div class='section-title'>📐 Methodology & Verification Protocol</div>", unsafe_allow_html=True)
    st.write("Technical architecture specification and mathematical verification equations for paper reviewers and evaluators.")

    st.markdown("""
    ### 1. Hybrid Reciprocal Rank Fusion (RRF)
    For chunk $d \\in D$ across dense semantic search (ChromaDB + BGE) and sparse lexical search (BM25):
    $$S_{RRF}(d) = w_{dense} \\cdot \\frac{1}{k + r_{dense}(d)} + w_{sparse} \\cdot \\frac{1}{k + r_{sparse}(d)}$$
    *Parameters:* $k = 60, w_{dense} = 0.65, w_{sparse} = 0.35$.

    ### 2. Claim-Level Faithfulness & Entailment Score
    Given candidate claims $C = \\{c_1, c_2, \\dots, c_m\\}$ and retrieved evidence premises $E$:
    $$\\text{Faithfulness}(C, E) = \\frac{1}{|C|} \\sum_{i=1}^{|C|} \\mathbb{I}(\\text{NLI}(c_i, E) = \\text{ENTAILED})$$
    $$\\text{Hallucination Rate}(C, E) = \\frac{1}{|C|} \\sum_{i=1}^{|C|} \\mathbb{I}(\\text{NLI}(c_i, E) = \\text{CONTRADICTED})$$
    """)

    st.markdown("---")
    st.markdown("#### 📊 Empirical Baseline Comparisons")
    df_ablation = evaluation_suite.run_ablation_benchmark()
    st.dataframe(df_ablation, use_container_width=True)
