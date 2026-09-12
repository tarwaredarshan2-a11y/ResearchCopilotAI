"""
utils package
-------------
Core utilities for the 6-Agent Verifiable Multimodal Research Co-Pilot:
- Multimodal PDF & document ingestion
- Hybrid vector & BM25 retrieval
- 6-Agent orchestration framework
- Claim & citation verification
- IEEE drafting
"""

from utils.multimodal_parser import parse_multimodal_pdf, chunk_parsed_document
from utils.hybrid_retriever import hybrid_retriever, HybridRetriever, BM25Retriever
from utils.orchestrator import orchestrator, ResearchCoPilotOrchestrator

__all__ = [
    "parse_multimodal_pdf",
    "chunk_parsed_document",
    "hybrid_retriever",
    "HybridRetriever",
    "BM25Retriever",
    "orchestrator",
    "ResearchCoPilotOrchestrator"
]
