# ResearchX: A Human-in-the-Loop Multi-Agent Architecture for Verifiable Literature Synthesis and IEEE Draft Generation

**Authors:** Student Researcher(s)  
**Affiliation:** Department of Computer Science & Engineering  
**Target Venue:** IEEE Conference / Capstone Research Project  

---

## Abstract

Synthesizing scientific literature across multiple papers remains a major bottleneck in academic research, often leading to ungrounded claims and missed research gaps. We introduce **ResearchX — Multi-Agent Research Intelligence**, an open human-in-the-loop framework that automates comparative literature analysis while preserving page-level citation provenance. ResearchX ingests PDF manuscripts via `PyMuPDF` and combines dense semantic search with BM25 lexical ranking using Reciprocal Rank Fusion (RRF). Retrieved context feeds into a six-agent orchestration workflow responsible for query decomposition, document retrieval, citation verification, claim entailment, research gap detection, and IEEE-compliant LaTeX draft generation. Beyond basic question answering, the framework extracts mathematical formulas, identifies cross-paper empirical conflicts, and compiles comparative literature matrices. Crucially, ResearchX functions as an advisory workspace—leaving final interpretation and synthesis decisions to human authors. We also propose a standardized evaluation protocol to assess retrieval quality, citation accuracy, claim entailment, and pipeline latency.

**Index Terms**—Retrieval-Augmented Generation (RAG), Multi-Agent Systems, Reciprocal Rank Fusion (RRF), Citation Grounding, Claim Verification, Literature Synthesis, IEEE Drafting.

---

## I. Introduction

The exponential expansion of academic literature across computer science, artificial intelligence, and software engineering presents researchers with a formidable challenge: staying abreast of domain advancements while systematically analyzing, contrasting, and synthesizing findings from dozens of primary papers. Manual literature review is inherently labor-intensive and prone to selection bias or missed research opportunities. 

While recent advances in Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) offer promising avenues for document interaction, standard single-pass LLM interfaces suffer from critical vulnerabilities when applied to scholarly analysis:
1. **Hallucinated Statements & Vague Attributions:** Standard generative models often produce plausible-sounding assertions without precise attribution to specific pages or passages.
2. **Lexical & Semantic Retrieval Blindspots:** Pure dense vector search frequently fails on exact keyword matching (e.g., specific hardware model numbers, software library names, or mathematical parameters), whereas pure lexical search fails on semantic rephrasing.
3. **Lack of Structure in Synthesis:** Simple chatbot interfaces lack dedicated workflows for cross-paper conflict resolution, quantitative parameter extraction, and formal academic drafting.

To address these limitations, this paper presents **ResearchX — Multi-Agent Research Intelligence**, a multi-agent human-in-the-loop software platform designed to assist researchers in analyzing, comparing, and drafting academic literature syntheses. Rather than positioning the system as an autonomous author, ResearchX operates as a verifiable research co-pilot that grounds every output in exact page-level citations (`[PaperName, p.X]`) and presents preliminary IEEE section drafts for human editing and validation.

### Major Technical Contributions
- **Hybrid Dense-Sparse RRF Retrieval Engine:** Combines dense embeddings (`BAAI/bge-small-en-v1.5`) via ChromaDB with custom BM25 sparse keyword ranking through Reciprocal Rank Fusion (RRF) to optimize document context retrieval across diverse query types.
- **Coordinated 6-Agent Pipeline:** Establishes a modular multi-agent workflow covering (1) Query Decomposition, (2) Literature Retrieval, (3) Evidence & Citation Verification, (4) LLM-Assisted Claim Verification, (5) Research-Gap Synthesis, and (6) IEEE Section & LaTeX Drafting.
- **Deep Academic Analysis Suite:** Provides dedicated modules for cross-paper empirical conflict detection, quantitative formula extraction with LaTeX formatting, structured literature review matrices, and automated IEEEtran LaTeX draft compilation.
- **Formal Evaluation Framework:** Establishes a systematic evaluation protocol defining metrics for retrieval precision/recall, citation grounding accuracy, premise-hypothesis claim entailment, and end-to-end pipeline latency.

---

## II. Related Work

### A. Retrieval-Augmented Generation in Scholarly Domains
Retrieval-Augmented Generation (RAG) enhances language models by grounding text generation in external knowledge stores. In academic document processing, standard RAG pipelines typically chunk PDF text into fixed-size character blocks, convert chunks into dense vector embeddings using pre-trained transformer models, and query a vector store using cosine similarity. However, dense retrieval alone often underperforms on scholarly texts rich in domain-specific terminology, specialized notation, and exact numeric thresholds.

### B. Hybrid Retrieval & Rank Fusion
To overcome dense retrieval limitations, hybrid retrieval mechanisms pair dense semantic embeddings with sparse lexical search methods like BM25. Reciprocal Rank Fusion (RRF) provides an effective parameter-free ensemble strategy that combines ranked lists from multiple search algorithms by computing an aggregated score based on reciprocal rank positions. ResearchX utilizes RRF to unify dense ChromaDB search results with BM25 keyword matches, ensuring both semantic depth and lexical precision.

### C. Multi-Agent Systems for Complex Reasoning
Decomposing complex tasks into specialized collaborative agents improves reasoning fidelity and reduces error propagation compared to monolithic LLM prompts. Existing multi-agent frameworks assign specialized roles (e.g., planner, retriever, critic, synthesizer). ResearchX adopts this paradigm by implementing a strictly scoped 6-agent pipeline where each agent enforces verifiable constraints before passing state down the execution graph.

---

## III. System Architecture & Multi-Agent Pipeline

ResearchX is built around a modular architecture comprising a layout-aware PDF ingestion pipeline, a hybrid vector-lexical retrieval engine, and a 6-agent execution workflow.

```
                                  [ User Research Query ]
                                             │
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │  Agent 1: Query Decomposition & Sub-Queries   │
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
                     │    Agent 5: Research-Gap Synthesizer Agent    │
                     │  (Cross-paper trajectory & gap matrix)        │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │  Agent 6: IEEE Section & LaTeX Drafting Agent │
                     │  (Synthesizes IEEEtran sections & LaTeX code) │
                     └───────────────────────────────────────────────┘
```

### A. Document Ingestion & Layout-Aware Parsing
Academic PDF documents are uploaded through the Streamlit user interface and parsed using `PyMuPDF` (`fitz`). The parser extracts document metadata, clean text streams, and precise page boundary markers (`page_number`). Extracted text is split into semantic chunks (default 600 characters with 100-character overlap) while preserving the originating PDF filename and page number within chunk metadata.

### B. Hybrid Retrieval Engine (Dense + Sparse RRF)
For every chunk, dense vector representations are generated using `BAAI/bge-small-en-v1.5` (384 dimensions) and stored in ChromaDB. Concurrently, tokenized representations are indexed into an inverted sparse index for custom BM25 ranking. 

When a search query $q$ is issued, both dense cosine similarity and sparse BM25 scores are calculated. The overall rank $RRF(d)$ for document chunk $d$ is computed using Reciprocal Rank Fusion:

$$RRF(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{k + r_m(d)}$$

where $r_m(d)$ represents the ordinal rank of chunk $d$ under retrieval system $m$, and $k$ is a constant scaling factor (set to $k = 60$).

### C. The Coordinated 6-Agent Execution Pipeline

1. **Agent 1 (Query Decomposition Agent):** Analyzes incoming research queries and decomposes broad questions into targeted atomic sub-queries focused on methodology, empirical results, dataset characteristics, and baseline comparisons.
2. **Agent 2 (Literature Retrieval Agent):** Executes hybrid RRF retrieval across all indexed documents for each sub-query, compiling a unified set of candidate context passages.
3. **Agent 3 (Evidence & Citation Verifier Agent):** Evaluates retrieved passages against source text metadata, enforcing page-level grounding and attaching structured bracketed provenance tags (`[PaperName, p.X]`).
4. **Agent 4 (LLM-Assisted Claim Verifier Agent):** Performs premise-hypothesis natural language entailment analysis on generated assertions against retrieved text chunks, categorizing claims as *Entailed*, *Contradicted*, or *Uncertain* with confidence scores.
5. **Agent 5 (Research-Gap Synthesizer Agent):** Analyzes verified evidence across papers to identify unaddressed limitations, conflicting methodology assumptions, and open research directions.
6. **Agent 6 (IEEE Section & LaTeX Drafting Agent):** Formulates structured manuscript draft sections aligned with standard IEEE conference guidelines (Abstract, Introduction, Related Work, Methodology, Results, Discussion, Conclusion) and emits compilable `IEEEtran` LaTeX source code.

---

## IV. Advanced System Capabilities

### A. Page-Level Provenance Grounding
To eliminate ungrounded output, all responses generated within the Continuous Chat and Literature Review modules mandate strict bracketed citation tags (`[PaperName, p.X]`). The system exposes the exact underlying source passage, allowing researchers to verify statements against the primary text instantly.

### B. Cross-Paper & Intra-Paper Empirical Conflict Detection
ResearchX includes a specialized conflict detection module that compares empirical statements, baseline metrics, and methodological assumptions across uploaded papers. The agent extracts pairs of assertions addressing identical evaluation criteria (e.g., model accuracy, memory overhead, latency thresholds) and flags contradictions for human review.

### C. Quantitative Formula Extraction & LaTeX Rendering
Recognizing that academic papers convey critical logic via mathematical notation, ResearchX parses mathematical expressions, hardware sampling frequencies, and dataset dimensions from PDF text streams. Extracted mathematical expressions are formatted into standard LaTeX equation blocks (`$$\dots$$`) for preview and verification.

### D. Literature Review Matrix & Gap Synthesis
The system constructs dynamic markdown comparative review tables structured as:

$$\text{Paper} \mid \text{Core Methodology} \mid \text{Empirical Dataset} \mid \text{Key Findings} \mid \text{Identified Limitations}$$

This structured synthesis enables researchers to compare multiple studies side-by-side and isolate unaddressed research gaps.

---

## V. Proposed Evaluation Methodology & Metrics

To evaluate ResearchX rigorously across diverse academic corpora, we outline a comprehensive evaluation protocol targeting four primary performance axes:

### A. Retrieval Quality Metrics
Evaluating the hybrid RRF engine against dense-only and sparse-only baselines across annotated benchmark queries:
- **Precision@K:** Fraction of retrieved passages in top-$K$ that contain ground-truth evidence.
- **Recall@K:** Fraction of total ground-truth evidence passages retrieved in top-$K$.
- **Mean Reciprocal Rank (MRR):** Reciprocal rank of the first relevant passage averaged over queries:
  $$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

### B. Citation Grounding Accuracy
- **Page Provenance Accuracy:** Percentage of generated citation tags (`[Paper, p.X]`) whose corresponding page contains the cited claim.
- **Attribution Precision:** Ratio of supported assertions to total cited assertions.

### C. Claim Verification Entailment
- **Entailment Accuracy:** Classification accuracy of Agent 4 on benchmark premise-hypothesis pairs (*Entailed*, *Contradicted*, *Uncertain*).
- **False Entailment Rate:** Percentage of ungrounded or contradictory statements incorrectly classified as entailed.

### D. System Latency & Overhead Analysis
- **Stage-wise Latency Breakdown:** Execution duration (in seconds) recorded across (1) PDF parsing, (2) Hybrid RRF retrieval, (3) Multi-agent LLM reasoning, and (4) IEEE draft compilation.

---

## VI. Discussion & Limitations

ResearchX is designed strictly as a **human-in-the-loop research co-pilot**. It does not produce unreviewed publication-ready manuscripts without author supervision. Key limitations of the current implementation include:
1. **Complex Multi-Column & Diagram Parsing:** While `PyMuPDF` reliably extracts clean text streams, embedded vector diagrams, complex nested tables, and non-standard two-column mathematical layouts can occasionally introduce text segmentation artifacts requiring manual author correction.
2. **LLM Context Window Boundaries:** Processing extensive multi-paper corpora requires strategic query decomposition to stay within prompt context windows during multi-agent synthesis.

---

## VII. Conclusion & Future Work

This paper presented **ResearchX — Multi-Agent Research Intelligence**, a multi-agent human-in-the-loop framework for scientific literature analysis and preliminary IEEE manuscript drafting. By integrating `PyMuPDF` PDF parsing, hybrid dense-sparse RRF retrieval, and a coordinated six-agent pipeline, ResearchX enables verifiable, page-grounded literature synthesis while keeping research interpretation firmly under human control. 

Future work will expand the platform by incorporating multimodal diagram-to-LaTeX conversion engines, automated citation graph analysis via Semantic Scholar APIs, and expanded support for ACM and Springer manuscript templates.

---

## References

1. J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.
2. P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, 2020, pp. 9459–9474.
3. S. Robertson, H. Zaragoza, et al., "The probabilistic relevance framework: BM25 and beyond," *Found. Trends Inf. Retrieval*, vol. 3, no. 4, pp. 333–389, 2009.
4. G. V. Cormack, C. L. A. Clarke, and S. Buettcher, "Reciprocal rank fusion outperforms data fusion combinations," in *Proc. ACM SIGIR Conf. Res. Dev. Inf. Retrieval*, 2009, pp. 742–743.
5. Q. Wu et al., "AutoGen: Enabling next-gen LLM applications via multi-agent conversation," *arXiv preprint arXiv:2308.08155*, 2023.
6. X. Wang et al., "Self-consistency improves chain of thought reasoning in language models," in *Proc. ICLR*, 2023.
7. IEEE Author Center, "IEEE Conference Paper Templates and Formatting Guidelines," IEEE, 2024. [Online]. Available: https://conferences.ieeeauthorcenter.ieee.org/
