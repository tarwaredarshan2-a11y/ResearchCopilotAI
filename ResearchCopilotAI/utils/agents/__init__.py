"""
agents package initialization
"""
from utils.agents.base_agent import BaseAgent
from utils.agents.query_decomposer import QueryDecompositionAgent
from utils.agents.lit_retriever import LiteratureRetrievalAgent
from utils.agents.citation_verifier import CitationVerificationAgent
from utils.agents.claim_verifier import ClaimVerificationAgent
from utils.agents.gap_synthesizer import ResearchGapSynthesisAgent
from utils.agents.ieee_drafter import IEEEDraftingAgent

__all__ = [
    "BaseAgent",
    "QueryDecompositionAgent",
    "LiteratureRetrievalAgent",
    "CitationVerificationAgent",
    "ClaimVerificationAgent",
    "ResearchGapSynthesisAgent",
    "IEEEDraftingAgent"
]
