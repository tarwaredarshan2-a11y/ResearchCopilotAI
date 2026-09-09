"""
config.py
---------
Central configuration module for ResearchCopilot AI (6-Agent Verifiable Multimodal Research Co-Pilot).

All shared constants, paths, agent hyperparameters, retrieval settings,
and IEEE formatting schemas are centralized here.
"""

import os
from dotenv import load_dotenv

# Base Directory & Explicit .env Loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_FILE):
    load_dotenv(dotenv_path=ENV_FILE)
else:
    load_dotenv()

# --------------------------------------------------------------------------
# API KEYS & MULTI-KEY SEQUENTIAL FAILOVER POOL
# --------------------------------------------------------------------------
raw_keys = os.getenv("GOOGLE_API_KEYS", "")
if raw_keys:
    GOOGLE_API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]
else:
    single_key = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_API_KEYS = [single_key.strip()] if single_key.strip() else []

# Fallback default key
GOOGLE_API_KEY = GOOGLE_API_KEYS[0] if GOOGLE_API_KEYS else ""

# --------------------------------------------------------------------------
# DIRECTORY PATHS
# --------------------------------------------------------------------------
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
DATA_DIR = os.path.join(BASE_DIR, "data")
BENCHMARK_DIR = os.path.join(DATA_DIR, "benchmarks")

# Ensure required directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# LLM & EMBEDDING SETTINGS
# --------------------------------------------------------------------------
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
LLM_TEMPERATURE = 0.2
LLM_MAX_OUTPUT_TOKENS = 4096

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_COLLECTION_NAME = "research_copilot_papers"

# --------------------------------------------------------------------------
# MULTIMODAL INGESTION & CHUNKING
# --------------------------------------------------------------------------
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
ENABLE_OCR_FALLBACK = True
MAX_PAGE_IMAGE_DPI = 200

# --------------------------------------------------------------------------
# HYBRID RETRIEVAL & RERANKING
# --------------------------------------------------------------------------
RETRIEVER_TOP_K = 6
HYBRID_DENSE_WEIGHT = 0.65
HYBRID_SPARSE_WEIGHT = 0.35
RRF_K = 60  # Reciprocal Rank Fusion smoothing constant

# --------------------------------------------------------------------------
# 6-AGENT CONFIGURATIONS
# --------------------------------------------------------------------------
AGENT_CONFIGS = {
    "agent1_query_decomposer": {
        "name": "Query Decomposition Agent",
        "description": "Deconstructs complex research queries into targeted sub-questions and search facets.",
        "temperature": 0.2,
    },
    "agent2_lit_retriever": {
        "name": "Literature Retrieval Agent",
        "description": "Executes hybrid (Dense + BM25) retrieval across academic papers with section filtering.",
        "temperature": 0.1,
    },
    "agent3_citation_verifier": {
        "name": "Evidence & Citation Verification Agent",
        "description": "Grounds claims with exact source snippets, page numbers, and validates citation authenticity.",
        "temperature": 0.0,
    },
    "agent4_claim_verifier": {
        "name": "Claim-Level Verification Agent",
        "description": "NLI-based claim validation (Entailment/Contradiction/Neutral) and hallucination filtering.",
        "temperature": 0.0,
        "entailment_threshold": 0.75,
    },
    "agent5_gap_synthesizer": {
        "name": "Research-Gap Synthesis Agent",
        "description": "Discovers cross-paper methodological limitations, missing datasets, and open questions.",
        "temperature": 0.3,
    },
    "agent6_ieee_drafter": {
        "name": "Drafting & IEEE Formatting Agent",
        "description": "Compiles verified synthesis into IEEE conference-compliant sections and BibTeX citations.",
        "temperature": 0.2,
    },
}

# --------------------------------------------------------------------------
# IEEE PAPER STRUCTURE SCHEMAS
# --------------------------------------------------------------------------
IEEE_SECTIONS = [
    "Abstract",
    "I. Introduction",
    "II. Related Work",
    "III. Methodology & System Architecture",
    "IV. Experimental Results & Verification",
    "V. Discussion & Research Gaps",
    "VI. Conclusion",
    "References",
]

# --------------------------------------------------------------------------
# APP UI BRANDING
# --------------------------------------------------------------------------
APP_TITLE = "Research Paper Co-Pilot"
APP_SUBTITLE = "6-Agent Verifiable Multimodal Research Assistant & IEEE Paper Synthesizer"
APP_ICON = "🔬"
