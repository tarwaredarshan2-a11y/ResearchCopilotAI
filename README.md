# 🔬 Research Paper Co-Pilot AI

**A Verifiable Multi-Agent Framework for Scientific Literature Synthesis & IEEE Conference Manuscript Generation**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit 1.36+](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-000000.svg?style=flat)](https://trychroma.com)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C.svg?style=flat)](https://langchain.com)
[![IEEE Format Aligned](https://img.shields.io/badge/IEEE-Author_Center_Compliant-006699.svg?style=flat)](https://conferences.ieeeauthorcenter.ieee.org/)

---

## 📌 1. Project Overview & Academic Purpose

**Research Paper Co-Pilot AI** is an autonomous scientific research assistant and publication synthesizer developed for 5th-semester engineering capstone evaluation and IEEE conference manuscript submissions. 

The platform transforms raw academic PDF documents into structured, grounded scientific knowledge. It addresses the key limitations of standard LLM summarizers—specifically ungrounded hallucinations, vague citation attribution, and lack of quantitative precision—by combining **layout-aware multimodal parsing**, **hybrid dense-sparse retrieval (Dense ChromaDB + Lexical BM25)**, **verified page-level provenance grounding**, and a **coordinated 6-agent orchestration pipeline**.

---

## ✨ 2. Key Verifiable Features

- **💬 Grounded AI Research Assistant**: Multi-turn continuous chat engine that answers questions strictly using retrieved paper passages with bracketed page-level citations (`[PaperName, p.X]`).
- **🔍 Document Analysis Teardown**: Extracts structured academic breakdowns (Abstract, Problem Statement, Methodology, Empirical Datasets, Key Findings, Limitations).
- **⚡ Cross-Paper & Intra-Paper Conflict Detector**: Identifies empirical contradictions, conflicting methodological assumptions, and internal trade-offs across single or multiple papers.
- **📐 Quantitative Setup & Formula Extractor**: Extracts equations, converts plain text formulas into formatted $\LaTeX$ equations, and parses hardware sensor parameters, sampling frequencies, and dataset metrics.
- **🧭 Literature Review & Research Gap Matrix**: Generates comparative literature review tables (`Paper | Methodology | Dataset | Key Results`) and detects unresolved research trajectories.
- **✍️ IEEE Manuscript & Overleaf LaTeX Studio**: Synthesizes camera-ready IEEE conference drafts and copy-pasteable **IEEEtran Overleaf LaTeX (`.tex`)** source code.
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
                     │  (Hybrid Dense ChromaDB + BM25 Lexical RRF)   │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │   Agent 3: Evidence & Citation Verifier       │
                     │  (Grounds claims to exact page quotes [p.X])  │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │    Agent 4: Claim-Level Verification Agent    │
                     │  (NLI-based Entailment vs Contradiction)      │
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
                     │  (Generates I-VI IEEE sections & Overleaf .tex)│
                     └───────────────────────────────────────────────┘
```

### Agent Roles & Specifications:
1. **Agent 1: Query Decomposer (`query_decomposer.py`)**: Deconstructs high-level queries into specific sub-questions and keyword search facets.
2. **Agent 2: Hybrid Literature Retriever (`lit_retriever.py`)**: Executes dense vector similarity search (ChromaDB) and sparse lexical search (BM25) with Reciprocal Rank Fusion.
3. **Agent 3: Evidence & Citation Verifier (`citation_verifier.py`)**: Binds generated statements to exact page-level text snippets and document provenance.
4. **Agent 4: Claim-Level Verifier (`claim_verifier.py`)**: Verifies candidate assertions against literature premises, categorizing claims into `ENTAILED`, `NEUTRAL`, or `CONTRADICTED`.
5. **Agent 5: Research-Gap Synthesizer (`gap_synthesizer.py`)**: Discovers methodological blindspots, missing evaluation benchmarks, and open research directions.
6. **Agent 6: IEEE Section & LaTeX Drafter (`ieee_drafter.py`)**: Formats syntheses into standardized IEEE conference paper sections, bracketed citations `[1]`, BibTeX entries, and Overleaf `.tex` source code.

---

## 🧮 4. Mathematical Formulation & Technical Implementation

### A. Hybrid Reciprocal Rank Fusion (RRF)
For a document passage $d \in D$, the hybrid score $S_{\text{RRF}}(d)$ combining dense vector ranking $r_{\text{dense}}(d)$ and sparse BM25 ranking $r_{\text{sparse}}(d)$ is calculated as:
$$S_{\text{RRF}}(d) = w_{\text{dense}} \cdot \frac{1}{k + r_{\text{dense}}(d)} + w_{\text{sparse}} \cdot \frac{1}{k + r_{\text{sparse}}(d)}$$

Where:
- $k = 60$ (Smoothing constant)
- $w_{\text{dense}} = 0.65$ (Dense vector similarity weight)
- $w_{\text{sparse}} = 0.35$ (Sparse lexical BM25 weight)

### B. Balanced Passage Sampling
To prevent a long document from dominating LLM context windows during multi-paper comparison, passages are sampled evenly across three document regions:
- **Beginning**: Abstract & Introduction ($0\% - 30\%$)
- **Middle**: Methodology & System Design ($30\% - 70\%$)
- **End**: Experimental Results & Discussion ($70\% - 100\%$)

---

## 🛠️ 5. Technology Stack & Dependencies

| Layer | Component / Library | Purpose |
| :--- | :--- | :--- |
| **Frontend / UI** | Streamlit (`>= 1.36.0`) | Responsive light-mode dashboard with periwinkle theme |
| **LLM Engine** | Google Gemini API (`gemini-1.5-flash` / `pro`) | Reasoning, claim verification, and LaTeX generation |
| **Vector Database** | ChromaDB (`langchain-chroma`) | Persistent vector storage for document embeddings |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) | Local CPU sentence embeddings |
| **Sparse Retrieval** | `rank_bm25` | Lexical keyword search |
| **PDF Extraction** | `pdfplumber`, `PyPDF2`, `fitz` | Text, page metadata, and layout parsing |
| **Evaluation Suite** | Python ROUGE / BLEU / Precision Metrics | Internal pipeline performance verification |

---

## 🚀 6. Installation & Execution Guide

### Prerequisites
- Python 3.10 or higher
- PowerShell / Terminal
- Google Gemini API Key

### Step 1: Clone & Navigate to Project Directory
```powershell
cd "ResearchCopilotAI"
```

### Step 2: Set Up Virtual Environment & Install Dependencies
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create a `.env` file inside the `ResearchCopilotAI/` folder:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
# Optional: Multi-key fallback pool
GOOGLE_API_KEYS=key1,key2,key3
```

### Step 4: Launch the Streamlit Application
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📄 7. IEEE Author Center Guidelines Compliance

All drafted IEEE manuscripts strictly conform to **[IEEE Author Center Conference Standards](https://conferences.ieeeauthorcenter.ieee.org/)**:
- **Title & Abstract**: Problem-Method-Result format with index terms.
- **Section I. Introduction**: Contextual background, problem statement, and enumerated contributions.
- **Section II. Related Work & Comparative Taxonomy**: Grounded review with bracketed citations `[1]`, `[2]`.
- **Section III. System Architecture & Methodology**: Pipeline design and mathematical formulations.
- **Section IV. Experimental Results**: Quantitative evaluation and benchmark comparison.
- **Section V. Discussion & Research Gaps**: Limitations and future directions.
- **Section VI. Conclusion**: Summary of contributions and closing remarks.
- **BibTeX & Overleaf Code**: Standard `.bib` entries and ready-to-compile `.tex` markup.

---

## 📁 8. Directory Structure

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
    ├── embeddings.py            # HuggingFace & ChromaDB interface
    ├── retriever.py             # Ordered chunk retrieval & balanced sampling
    ├── hybrid_retriever.py      # Dense + BM25 + RRF Hybrid Retriever
    ├── llm.py                   # Multi-key Gemini API failover pool
    ├── paper_analysis.py        # Single & multi-paper document teardown
    ├── conflict_detector.py     # Cross-paper contradiction & risk detector
    ├── formula_extractor.py     # LaTeX equation & quantitative metric extractor
    ├── literature_review.py     # Comparative literature review generator
    ├── research_gap.py          # Cross-paper research gap detector
    ├── orchestrator.py          # 6-Agent pipeline orchestrator
    ├── evaluation_suite.py      # Metric evaluation runner
    └── agents/                  # Autonomous Agent implementations
        ├── base_agent.py        # Base Agent interface
        ├── query_decomposer.py  # Agent 1: Query Decomposition
        ├── lit_retriever.py     # Agent 2: Literature Retrieval
        ├── citation_verifier.py # Agent 3: Citation Grounding
        ├── claim_verifier.py    # Agent 4: Claim-Level Verification
        ├── gap_synthesizer.py   # Agent 5: Research Gap Synthesis
        └── ieee_drafter.py      # Agent 6: IEEE Manuscript & LaTeX Drafter
```

---

## 📜 9. License & Academic Attribution

Developed for **5th Semester Engineering Capstone Project** and **IEEE Conference Publication Synthesis**.
© 2026 Research Co-Pilot AI • All Rights Reserved.
