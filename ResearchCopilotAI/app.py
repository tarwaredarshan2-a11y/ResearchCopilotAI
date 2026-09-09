"""
app.py
------
Research Paper Co-Pilot — 6-Agent Verifiable Multimodal Research Assistant.

Features clean separation between:
1. 🎓 Researcher Workspace (User View): Paper Library, AI Chat with Grounded Citations, Deep Paper Analysis, Cross-Paper Literature Review & Gaps, IEEE Paper Drafter.
2. ⚙️ Developer & Benchmark Suite (Developer View): 6-Agent Tracing, Claim-Level NLI Matrix, Ablation Benchmarks, Hyperparameter Tuning & Key Pool Diagnostics.
"""

import os
import streamlit as st
import pandas as pd
import json

from config import (
    APP_TITLE, APP_SUBTITLE, APP_ICON, GOOGLE_API_KEY, GOOGLE_API_KEYS,
    RETRIEVER_TOP_K, HYBRID_DENSE_WEIGHT, HYBRID_SPARSE_WEIGHT,
    AGENT_CONFIGS, IEEE_SECTIONS
)
from utils.pdf_loader import save_uploaded_pdf, get_pdf_page_count
from utils.multimodal_parser import parse_multimodal_pdf, chunk_parsed_document
from utils.embeddings import add_documents_to_vector_store, get_all_paper_names
from utils.hybrid_retriever import hybrid_retriever
from utils.orchestrator import orchestrator
from utils.evaluation_suite import evaluation_suite
from utils.llm import generate_response, key_manager
from utils.paper_analysis import analyze_paper, ANALYSIS_FIELDS
from utils.literature_review import generate_literature_review
from utils.research_gap import detect_research_gaps

# ==========================================================================
# PAGE CONFIG & MODERN ACADEMIC STYLING
# ==========================================================================
st.set_page_config(
    page_title=f"{APP_TITLE} | Verifiable Academic AI",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    .user-card {
        background: linear-gradient(135deg, #161b22, #1c2128);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .agent-trace-card {
        background: #161b22;
        border-left: 4px solid #58a6ff;
        border-radius: 4px 10px 10px 4px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .metric-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-entailed { background-color: #1a472a; color: #7ee787; border: 1px solid #2ea043; }
    .badge-neutral { background-color: #4d3800; color: #e3b341; border: 1px solid #9e6a03; }
    .badge-contradicted { background-color: #4c1d1d; color: #f85149; border: 1px solid #da3633; }
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #58a6ff;
        border-bottom: 2px solid #21262d;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    .quote-box {
        border-left: 3px solid #79c0ff;
        background-color: #13171e;
        padding: 10px 14px;
        font-style: italic;
        margin: 8px 0;
        border-radius: 0 6px 6px 0;
    }
    .chat-bubble-user {
        background-color: #1f242c;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .chat-bubble-ai {
        background-color: #161b22;
        border: 1px solid #388bfd33;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_pipeline_result" not in st.session_state:
    st.session_state.last_pipeline_result = None

# ==========================================================================
# SIDEBAR: MODE SWITCH & ENVIRONMENT STATUS
# ==========================================================================
with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption("Verifiable 6-Agent Academic Synthesis Framework")
    st.markdown("---")

    # PRIMARY VIEW SWITCH (User vs Developer Mode)
    app_mode = st.radio(
        "🎛️ **Select Interface View:**",
        ["🎓 Researcher Workspace (User View)", "⚙️ Developer & Benchmark Suite"],
        index=0
    )
    st.markdown("---")

    paper_names = get_all_paper_names()
    st.metric("📚 Indexed Papers", len(paper_names))

    # Multi-Key Pool Info (Compact for User, Detailed for Dev)
    key_stat = key_manager.get_status()
    if app_mode.startswith("⚙️"):
        st.subheader("🔑 API Key Pool Diagnostics")
        st.success(f"Pool: **{key_stat['total_keys']} Keys Active**")
        st.caption(f"Active Key: **#{key_stat['active_key_number']}** (`{key_stat['active_masked_key']}`)")
        st.caption("⚡ *Automatic sequential failover on quota / 429 limits.*")

        st.subheader("🎚️ Model Hyperparameters")
        dense_weight = st.slider("Dense Vector Weight", 0.0, 1.0, HYBRID_DENSE_WEIGHT, 0.05)
        top_k = st.slider("Retriever Top-K Chunks", 2, 12, RETRIEVER_TOP_K)
        claim_threshold = st.slider("Claim Entailment Threshold", 0.5, 0.95, 0.75, 0.05)
    else:
        st.success("🟢 AI Engine Online (Gemini Multi-Key Active)")
        st.info("💡 **IEEE Standard**: Drafted papers conform to IEEE Author Center conference format.")

# ==========================================================================
# VIEW 1: 🎓 RESEARCHER WORKSPACE (USER VIEW)
# ==========================================================================
if app_mode.startswith("🎓"):
    user_tabs = st.tabs([
        "📚 Paper Library & Ingest",
        "💬 Grounded AI Research Chat",
        "🔍 Single-Paper Deep Dive",
        "🧭 Literature Review & Gaps",
        "📄 IEEE Paper Drafter"
    ])

    # --- TAB 1: PAPER LIBRARY & INGEST ---
    with user_tabs[0]:
        st.markdown("<div class='section-title'>📚 Paper Library & Document Ingestion</div>", unsafe_allow_html=True)
        st.write("Upload PDF research papers to automatically parse text, sections, and tables into the research knowledge base.")

        col_u1, col_u2 = st.columns([2, 1])
        with col_u1:
            uploaded_files = st.file_uploader("Upload Academic Papers (PDF)", type=["pdf"], accept_multiple_files=True)
            if uploaded_files:
                if st.button("🚀 Process & Ingest Papers", type="primary"):
                    with st.spinner("Extracting layout, chunking, and indexing into vector store..."):
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

        with col_u2:
            st.markdown("#### 📑 Indexed Research Papers")
            if paper_names:
                for p in paper_names:
                    st.markdown(f"- 📄 **{p}**")
            else:
                st.info("No papers added yet. Upload a PDF on the left.")

    # --- TAB 2: GROUNDED AI RESEARCH CHAT ---
    with user_tabs[1]:
        st.markdown("<div class='section-title'>💬 Grounded AI Research Assistant</div>", unsafe_allow_html=True)
        st.write("Ask questions across your uploaded research papers. Every answer includes verifiable source citations and page numbers.")

        col_c1, col_c2 = st.columns([3, 1])
        with col_c1:
            user_question = st.text_input("Ask a research question:", placeholder="e.g. What datasets and evaluation metrics are used in these papers?")
        with col_c2:
            target_filter = st.selectbox("Focus on Paper:", ["All Papers"] + paper_names)
            filter_paper_name = None if target_filter == "All Papers" else target_filter

        if st.button("🔍 Search & Answer", type="primary") and user_question:
            with st.spinner("Retrieving literature passages and verifying citations..."):
                retrieved_chunks = hybrid_retriever.retrieve(
                    query=user_question,
                    top_k=5,
                    paper_name=filter_paper_name
                )
                
                context_str = "\n\n".join([
                    f"[Source: {c.get('metadata', {}).get('paper_name')} | Page: {c.get('metadata', {}).get('page_number')}]\n{c.get('text')}"
                    for c in retrieved_chunks
                ])

                chat_prompt = f"""You are an expert research assistant. Answer the user's question using ONLY the provided literature context.
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

        # Display Chat Feed
        for entry in reversed(st.session_state.chat_history):
            st.markdown(f"<div class='chat-bubble-user'><b>👤 Question:</b> {entry['question']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble-ai'><b>🧠 Assistant:</b><br><br>{entry['answer']}</div>", unsafe_allow_html=True)
            with st.expander("🔎 View Retrieved Citations & Quotes", expanded=False):
                for idx, src in enumerate(entry["sources"]):
                    meta = src.get("metadata", {})
                    st.markdown(f"**[{idx+1}] {meta.get('paper_name')}** (Page {meta.get('page_number')}, Section: {meta.get('section', 'General')})")
                    st.markdown(f"<div class='quote-box'>\"{src.get('text')[:250]}...\"</div>", unsafe_allow_html=True)
            st.markdown("---")

    # --- TAB 3: SINGLE-PAPER DEEP DIVE ---
    with user_tabs[2]:
        st.markdown("<div class='section-title'>🔍 Single-Paper Structured Analysis</div>", unsafe_allow_html=True)
        st.write("Extract a comprehensive academic breakdown of a specific paper (Title, Authors, Abstract, Methodology, Findings).")

        if paper_names:
            selected_analysis_paper = st.selectbox("Select Paper for Analysis:", paper_names)
            if st.button("📊 Run Deep Paper Analysis", type="primary"):
                with st.spinner("Analyzing methodology, datasets, and conclusions..."):
                    analysis_res = analyze_paper(selected_analysis_paper)
                    st.markdown(f"### 📄 Analysis of `{selected_analysis_paper}`")
                    for field, content in analysis_res.items():
                        with st.expander(f"📌 {field.replace('_', ' ').title()}", expanded=True):
                            st.write(content)
        else:
            st.info("Upload papers in the Library tab first.")

    # --- TAB 4: LITERATURE REVIEW & GAPS ---
    with user_tabs[3]:
        st.markdown("<div class='section-title'>🧭 Cross-Paper Literature Review & Research Gaps</div>", unsafe_allow_html=True)
        st.write("Synthesize cross-paper comparisons, find conflicting results, and discover open research opportunities.")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### 📚 Comprehensive Literature Review")
            if st.button("Generate Multi-Paper Review"):
                with st.spinner("Synthesizing multi-paper review..."):
                    rev = generate_literature_review(paper_names)
                    for k, v in rev.items():
                        st.markdown(f"**{k.replace('_', ' ').title()}**")
                        st.write(v)

        with col_r2:
            st.markdown("#### 🔬 Research Gap Matrix")
            if st.button("Discover Research Gaps"):
                with st.spinner("Identifying open challenges and limitations..."):
                    gaps = detect_research_gaps(paper_names)
                    for k, v in gaps.items():
                        st.markdown(f"**{k.replace('_', ' ').title()}**")
                        st.info(v)

    # --- TAB 5: IEEE PAPER DRAFTER ---
    with user_tabs[4]:
        st.markdown("<div class='section-title'>📄 Autonomous IEEE Paper Drafter</div>", unsafe_allow_html=True)
        st.write("Generate a publication-ready IEEE conference manuscript based on verified findings.")

        paper_topic = st.text_input("Enter Paper Focus / Title Idea:", value="A Verifiable Multi-Agent Framework for Scientific Literature Synthesis")
        
        if st.button("📝 Draft Complete IEEE Paper", type="primary"):
            with st.spinner("Executing 6-Agent pipeline to draft IEEE manuscript..."):
                pipe_res = orchestrator.run_full_pipeline(research_query=paper_topic)
                st.session_state.last_pipeline_result = pipe_res
                st.success("Draft generated successfully!")

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

            st.markdown("### 📑 BibTeX")
            st.code(a6.get("bibtex_entries", ""), language="bibtex")

            # Download Option
            formatted_refs = "\n".join(ref_list)
            full_md = f"# {a6.get('paper_title')}\n\n**Abstract**— {a6.get('abstract')}\n\n**Keywords**— {kw_str}\n\n## I. INTRODUCTION\n{sec.get('introduction')}\n\n## II. RELATED WORK\n{sec.get('related_work')}\n\n## III. METHODOLOGY\n{sec.get('methodology')}\n\n## IV. RESULTS\n{sec.get('experiments_and_results')}\n\n## V. DISCUSSION & GAPS\n{sec.get('discussion_and_gaps')}\n\n## VI. CONCLUSION\n{sec.get('conclusion')}\n\n## REFERENCES\n{formatted_refs}\n\n## BIBTEX\n```bibtex\n{a6.get('bibtex_entries')}\n```\n"
            st.download_button("📥 Download IEEE Paper Draft (.md)", data=full_md, file_name="IEEE_Manuscript_Draft.md", mime="text/markdown")

# ==========================================================================
# VIEW 2: ⚙️ DEVELOPER & BENCHMARK SUITE (DEVELOPER VIEW)
# ==========================================================================
else:
    dev_tabs = st.tabs([
        "🤖 6-Agent Live Traces",
        "✅ Claim-Level NLI Matrix",
        "🧪 Ablation & Evaluation Benchmarks",
        "📐 System Math & Equations"
    ])

    # --- DEV TAB 1: 6-AGENT TRACES ---
    with dev_tabs[0]:
        st.markdown("<div class='section-title'>🤖 6-Agent Live Pipeline Tracing & Inspectability</div>", unsafe_allow_html=True)
        st.write("Inspect individual agent outputs, intermediate JSON states, and sub-queries.")

        res = st.session_state.last_pipeline_result
        if res:
            with st.expander("Agent 1: Query Decomposition (JSON Trace)", expanded=True):
                st.json(res.get("agent1_output", {}))
            with st.expander("Agent 2: Literature Retrieval (Ranked Chunks)", expanded=False):
                st.json(res.get("agent2_output", {}))
            with st.expander("Agent 3: Evidence & Citation Grounding", expanded=False):
                st.json(res.get("agent3_output", {}))
            with st.expander("Agent 4: Claim NLI & Hallucination Filter", expanded=False):
                st.json(res.get("agent4_output", {}))
            with st.expander("Agent 5: Research-Gap Synthesis", expanded=False):
                st.json(res.get("agent5_output", {}))
            with st.expander("Agent 6: IEEE Drafter Output", expanded=False):
                st.json(res.get("agent6_output", {}))
        else:
            st.info("Run a query or draft a paper in Researcher View to generate agent trace logs.")

    # --- DEV TAB 2: CLAIM NLI MATRIX ---
    with dev_tabs[1]:
        st.markdown("<div class='section-title'>✅ Claim-Level NLI Verification Matrix</div>", unsafe_allow_html=True)
        st.write("Examines hypothesis-premise entailment verdicts (`ENTAILED`, `NEUTRAL`, `CONTRADICTED`) and confidence scores.")

        res = st.session_state.last_pipeline_result
        if res and "agent4_output" in res:
            a4 = res["agent4_output"]
            claims = a4.get("verified_claims", [])

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Faithfulness Score", f"{a4.get('overall_faithfulness_score', 0.9)*100:.1f}%")
            with c2:
                st.metric("Hallucination Rate", f"{a4.get('hallucination_rate', 0.05)*100:.1f}%")
            with c3:
                st.metric("Total Claims Tested", len(claims))

            claim_rows = []
            for c in claims:
                claim_rows.append({
                    "Claim ID": c.get("claim_id"),
                    "Claim Statement": c.get("claim_text"),
                    "Verdict": c.get("verdict"),
                    "Confidence": f"{c.get('confidence_score', 0.9):.2f}",
                    "Evidence IDs": ", ".join(c.get("supporting_evidence_ids", [])),
                    "Hallucination Risk": "High" if c.get("is_hallucination") else "Low"
                })
            st.dataframe(pd.DataFrame(claim_rows), use_container_width=True)
        else:
            st.info("No active claim data. Execute a research query to populate the NLI matrix.")

    # --- DEV TAB 3: ABLATION BENCHMARKS ---
    with dev_tabs[2]:
        st.markdown("<div class='section-title'>🧪 Academic Evaluation & Ablation Study Benchmarks</div>", unsafe_allow_html=True)
        st.write("Quantitative experimental metrics comparing naive dense RAG, sparse BM25, hybrid retrieval, and our full 6-agent verifiable framework.")

        df_ablation = evaluation_suite.run_ablation_benchmark()
        st.dataframe(df_ablation, use_container_width=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Faithfulness (%) by Architecture**")
            st.bar_chart(df_ablation.set_index("Configuration")["Faithfulness (%)"])
        with col_c2:
            st.markdown("**Hallucination Reduction (%)**")
            st.bar_chart(df_ablation.set_index("Configuration")["Hallucination Rate (%)"])

    # --- DEV TAB 4: SYSTEM MATH ---
    with dev_tabs[3]:
        st.markdown("<div class='section-title'>📐 Mathematical Formulations & Research Protocols</div>", unsafe_allow_html=True)
        st.markdown("""
        ### 1. Hybrid Reciprocal Rank Fusion (RRF)
        For chunk $d \\in D$ across dense and sparse retrieval:
        $$S_{RRF}(d) = w_{dense} \\cdot \\frac{1}{k + r_{dense}(d)} + w_{sparse} \\cdot \\frac{1}{k + r_{sparse}(d)}$$
        *Current parameters:* $k = 60, w_{dense} = 0.65, w_{sparse} = 0.35$.

        ### 2. Claim-Level Faithfulness & Hallucination Formulations
        $$\\text{Faithfulness}(C, E) = \\frac{1}{|C|} \\sum_{i=1}^{|C|} \\mathbb{I}(\\text{NLI}(c_i, E) = \\text{ENTAILED})$$
        $$\\text{Hallucination Rate}(C, E) = \\frac{1}{|C|} \\sum_{i=1}^{|C|} \\mathbb{I}(\\text{NLI}(c_i, E) = \\text{CONTRADICTED})$$
        """)
