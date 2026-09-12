# 🔬 Research Paper Co-Pilot AI

> **Human-in-the-loop framework for evidence-grounded multi-paper literature analysis and IEEE-structured draft generation.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit 1.36+](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-000000.svg?style=flat)](https://trychroma.com)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C.svg?style=flat)](https://langchain.com)
[![IEEE Structure Aligned](https://img.shields.io/badge/IEEE-Structure_Aligned-006699.svg?style=flat)](https://conferences.ieeeauthorcenter.ieee.org/)

---

## 📌 1. Project Overview & Academic Scope

**Research Paper Co-Pilot AI** is a human-in-the-loop research assistant and publication helper developed for 5th-semester engineering capstone evaluation and academic literature analysis.

The system assists researchers in ingesting, comparing, and synthesizing academic PDF documents. It addresses key challenges in automated literature review—such as ungrounded statements, vague citation attributions, and manual synthesis bottlenecks—by combining **layout-aware PDF text and metadata extraction (`PyMuPDF`)**, **hybrid dense-sparse retrieval (Dense ChromaDB + Custom Lexical BM25)**, **page-level provenance grounding**, and a **coordinated 6-agent orchestration pipeline**.

---

## ✨ 2. Key Functional Features

- **💬 Grounded AI Research Assistant**: Multi-turn continuous chat engine that answers questions using retrieved paper passages with bracketed page-level citations (`[PaperName, p.X]`).
- **🔍 Document Analysis Teardown**: Extracts structured academic breakdowns (Abstract, Problem Statement, Methodology, Empirical Datasets, Key Findings, Limitations).
- **⚡ Cross-Paper & Intra-Paper Conflict Detector**: Identifies empirical contradictions, conflicting methodological assumptions, and internal trade-offs across single or multiple papers.
- **📐 Quantitative Setup & Formula Extractor**: Extracts mathematical expressions and generates $\LaTeX$ representations for review, parsing hardware sensor parameters, sampling frequencies, and dataset metrics.
- **🧭 Literature Review & Research Gap Matrix**: Generates comparative literature review tables (`Paper | Methodology | Dataset | Key Results`) and identifies open research trajectories.
- **✍️ IEEE Manuscript & Overleaf LaTeX Studio**: Synthesizes structured IEEE conference draft sections and **IEEEtran-formatted LaTeX draft source requiring author and citation review**.
- **🗑️ Permanent Paper Storage Manager**: Allows users to select and permanently delete uploaded PDFs from disk storage and the Chroma vector database.

---

## 🏗️ 3. System Architecture & 6-Agent Pipeline

```
                                  [ User Research Query ]
                                             │
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │  Agent 1: Query Decomposition & Sub-Queries   │
                     │  (Breaks query into targeted sub-questions)   │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │     Agent 2: Literature Retrieval Agent       │
                     │  (Hybrid BGE-small + Custom BM25 RRF)         │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │   Agent 3: Evidence & Citation Verifier       │
                     │  (Grounds claims to page-level text [p.X])    │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │    Agent 4: LLM-Assisted Claim Verifier       │
                     │  (Premise-Hypothesis Entailment Check)        │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │     Agent 5: Research-Gap Synthesizer         │
                     │  (Cross-paper limitations & open questions)   │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │     Agent 6: IEEE Section & LaTeX Drafter     │
                     │  (Generates I-VI IEEE sections & LaTeX draft) │
                     └───────────────────────────────────────────────┘
```

### Agent Roles & Specifications:
1. **Agent 1: Query Decomposer (`query_decomposer.py`)**: Deconstructs high-level queries into specific sub-questions and keyword search facets.
2. **Agent 2: Hybrid Literature Retriever (`lit_retriever.py`)**: Executes dense vector similarity search (`BAAI/bge-small-en-v1.5`) and custom sparse lexical search (BM25) with Reciprocal Rank Fusion.
3. **Agent 3: Evidence & Citation Verifier (`citation_verifier.py`)**: Associates generated evidence statements with retrieved page-level source context.
4. **Agent 4: LLM-Assisted Claim Verifier (`claim_verifier.py`)**: Verifies candidate assertions against retrieved literature premises, categorizing claims into `ENTAILED`, `NEUTRAL`, or `CONTRADICTED`.
5. **Agent 5: Research-Gap Synthesizer (`gap_synthesizer.py`)**: Discovers methodological blindspots, missing evaluation benchmarks, and open research directions.
6. **Agent 6: IEEE Section & LaTeX Drafter (`ieee_drafter.py`)**: Formats syntheses into standardized IEEE conference paper sections, bracketed citations `[1]`, BibTeX entries, and draft LaTeX source code.

---

## 🧮 4. Mathematical Formulation & Technical Specifications

### A. Hybrid Reciprocal Rank Fusion (RRF)
For a document passage $d \in D$, the hybrid score $S_{\text{RRF}}(d)$ combining dense vector ranking $r_{\text{dense}}(d)$ and custom sparse BM25 ranking $r_{\text{sparse}}(d)$ is calculated as:
$$S_{\text{RRF}}(d) = w_{\text{dense}} \cdot \frac{1}{k + r_{\text{dense}}(d)} + w_{\text{sparse}} \cdot \frac{1}{k + r_{\text{sparse}}(d)}$$

Where:
- $k = 60$ (Smoothing constant)
- $w_{\text{dense}} = 0.65$ (Dense vector similarity weight)
- $w_{\text{sparse}} = 0.35$ (Sparse lexical BM25 weight)

### B. Balanced Passage Sampling
Chunks are selected at evenly spaced positions across each document to reduce the dominance of long papers in the context window during multi-paper comparative analysis.

---

## 🔬 5. Planned Evaluation Methodology

> **Note:** Evaluation experiments are planned using manually annotated research-paper queries and evidence passages.

The evaluation workflow will measure:
- **Retrieval Quality**: Precision@5, Recall@5, Mean Reciprocal Rank (MRR)
- **Factuality & Citation Grounding**: Citation correctness rate, claim entailment accuracy, unsupported-claim rate
- **System Performance**: Average query latency (seconds)

### Baseline Comparison Strategy:
1. Dense Retrieval Only (`BAAI/bge-small-en-v1.5`)
2. Sparse Lexical Retrieval Only (Custom BM25)
3. Hybrid Retrieval (Dense + BM25 RRF)
4. Hybrid Retrieval + Citation Grounding (Agents 1–3)
5. Full 6-Agent Pipeline (Agents 1–6)

---

## 🛠️ 6. System Reproducibility & Exact Configuration

| Parameter | Configuration Value | Location |
| :--- | :--- | :--- |
| **Python Version** | `Python >= 3.10` | Environment |
| **UI Framework** | `Streamlit >= 1.36.0` | `app.py` |
| **LLM Model** | `gemini-2.5-flash` | `config.py` |
| **Embedding Model** | `BAAI/bge-small-en-v1.5` | `config.py` |
| **PDF Parser** | `PyMuPDF (fitz)` | `utils/multimodal_parser.py` |
| **Vector Database** | ChromaDB (`langchain-chroma`) | `utils/embeddings.py` |
| **Chunk Size & Overlap** | $900$ characters / $150$ character overlap | `config.py` |
| **RRF Weights** | $w_{\text{dense}} = 0.65$, $w_{\text{sparse}} = 0.35$, $k = 60$ | `config.py` |

---

## 🚀 7. Installation & Setup Guide

### Step 1: Navigate to Project Directory
```powershell
cd "ResearchCopilotAI"
```

### Step 2: Create Environment & Install Dependencies
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 3: Configure Gemini API Key
Create a `.env` file inside `ResearchCopilotAI/`:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Step 4: Run the Streamlit Application
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## ⚠️ 8. System Limitations & Human-in-the-Loop Scope

- **Scanned PDFs**: Poorly scanned or low-resolution PDFs may have incomplete text extraction.
- **Complex Equations & Figures**: Multi-column inline equations and embedded figure diagrams require human review.
- **Probabilistic Verification**: LLM-assisted claim verification is probabilistic; generated citations must be verified against primary source documents.
- **Human Editing Required**: Generated IEEE drafts serve as preliminary literature reviews and require human editing prior to conference submission.
- **Publication Guarantee**: System outputs do not guarantee IEEE conference acceptance.

---

## 📁 9. Directory Structure

```
ResearchCopilotAI/
├── app.py                       # Modern 6-Tab Streamlit Web Application
├── config.py                    # Shared configuration & hyperparameters
├── requirements.txt             # Project Python dependencies
├── .env.example                 # Environment variables template
├── README.md                    # Verification documentation
├── uploads/                     # Ingested PDF research paper storage
├── vector_db/                   # Persistent ChromaDB vector database
└── utils/
    ├── __init__.py
    ├── pdf_loader.py            # PDF storage helper
    ├── multimodal_parser.py     # Layout-aware PDF text parser & chunker
    ├── embeddings.py            # HuggingFace BGE & ChromaDB interface
    ├── retriever.py             # Ordered chunk retrieval & balanced sampling
    ├── hybrid_retriever.py      # Dense + Custom BM25 + RRF Hybrid Retriever
    ├── llm.py                   # Multi-key Gemini API failover pool
    ├── paper_analysis.py        # Single & multi-paper document teardown
    ├── conflict_detector.py     # Cross-paper contradiction & risk detector
    ├── formula_extractor.py     # LaTeX equation & quantitative metric extractor
    ├── literature_review.py     # Comparative literature review generator
    ├── research_gap.py          # Cross-paper research gap detector
    ├── orchestrator.py          # 6-Agent pipeline orchestrator
    ├── evaluation_suite.py      # Metric evaluation runner
    └── agents/                  # Agent implementations
        ├── base_agent.py        # Base Agent interface
        ├── query_decomposer.py  # Agent 1: Query Decomposition
        ├── lit_retriever.py     # Agent 2: Literature Retrieval
        ├── citation_verifier.py # Agent 3: Citation Grounding
        ├── claim_verifier.py    # Agent 4: LLM Claim Verification
        ├── gap_synthesizer.py   # Agent 5: Research Gap Synthesis
        └── ieee_drafter.py      # Agent 6: IEEE Section & LaTeX Drafter
```

---

## 📜 10. License & Academic Attribution

This project is an academic capstone prototype. See the repository owner for usage and redistribution permissions.
© 2026 Research Co-Pilot AI.
