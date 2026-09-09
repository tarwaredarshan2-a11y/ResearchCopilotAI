"""
orchestrator.py
---------------
6-Agent Research Co-Pilot Workflow Orchestrator.

Manages pipeline state, orchestrates sequential agent execution,
and provides step-by-step progress tracking for the user interface.
"""

from typing import Dict, Any, Optional, Callable, List
from utils.agents.query_decomposer import QueryDecompositionAgent
from utils.agents.lit_retriever import LiteratureRetrievalAgent
from utils.agents.citation_verifier import CitationVerificationAgent
from utils.agents.claim_verifier import ClaimVerificationAgent
from utils.agents.gap_synthesizer import ResearchGapSynthesisAgent
from utils.agents.ieee_drafter import IEEEDraftingAgent

class ResearchCoPilotOrchestrator:
    def __init__(self):
        self.agent1_decomposer = QueryDecompositionAgent()
        self.agent2_retriever = LiteratureRetrievalAgent()
        self.agent3_citation_verifier = CitationVerificationAgent()
        self.agent4_claim_verifier = ClaimVerificationAgent()
        self.agent5_gap_synthesizer = ResearchGapSynthesisAgent()
        self.agent6_ieee_drafter = IEEEDraftingAgent()

    def run_full_pipeline(
        self,
        research_query: str,
        selected_paper: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete 6-Agent research pipeline end-to-end.
        """
        state: Dict[str, Any] = {
            "query": research_query,
            "selected_paper": selected_paper,
            "agent1_output": {},
            "agent2_output": {},
            "agent3_output": {},
            "agent4_output": {},
            "agent5_output": {},
            "agent6_output": {},
            "status": "in_progress"
        }

        # Step 1: Query Decomposition
        if progress_callback:
            progress_callback("Agent 1: Decomposing research query into structured sub-facets...", 15)
        decomp = self.agent1_decomposer.decompose(research_query)
        state["agent1_output"] = decomp

        # Step 2: Literature Retrieval
        if progress_callback:
            progress_callback("Agent 2: Performing Hybrid RAG (Dense + BM25) retrieval...", 35)
        retrieved = self.agent2_retriever.retrieve_literature(
            decomposition=decomp,
            selected_paper=selected_paper
        )
        state["agent2_output"] = retrieved

        # Step 3: Citation Verification & Grounding
        if progress_callback:
            progress_callback("Agent 3: Validating evidence & grounding citations...", 55)
        citation_data = self.agent3_citation_verifier.verify_citations(
            query_intent=decomp.get("core_research_intent", research_query),
            retrieved_data=retrieved
        )
        state["agent3_output"] = citation_data

        # Step 4: Claim-Level Verification
        if progress_callback:
            progress_callback("Agent 4: Executing NLI claim-level entailment & hallucination filtering...", 70)
        
        candidate_claims = [
            e.get("statement", "") for e in citation_data.get("grounded_evidence", [])
        ]
        if not candidate_claims:
            candidate_claims = [research_query]

        claim_data = self.agent4_claim_verifier.verify_claims(
            candidate_claims=candidate_claims,
            evidence_data=citation_data
        )
        state["agent4_output"] = claim_data

        # Step 5: Research-Gap Synthesis
        if progress_callback:
            progress_callback("Agent 5: Synthesizing cross-paper limitations & open research gaps...", 85)
        gap_data = self.agent5_gap_synthesizer.synthesize_gaps(
            research_topic=research_query,
            verified_evidence=citation_data.get("grounded_evidence", [])
        )
        state["agent5_output"] = gap_data

        # Step 6: IEEE Paper Section Drafting
        if progress_callback:
            progress_callback("Agent 6: Drafting IEEE conference-aligned sections & BibTeX...", 95)
        ieee_draft = self.agent6_ieee_drafter.draft_ieee_paper(
            topic=research_query,
            verified_evidence=citation_data.get("grounded_evidence", []),
            verified_claims=claim_data.get("verified_claims", []),
            research_gaps=gap_data.get("research_gaps", [])
        )
        state["agent6_output"] = ieee_draft

        state["status"] = "completed"
        if progress_callback:
            progress_callback("All 6 Agents completed synthesis successfully!", 100)

        return state

# Global singleton orchestrator
orchestrator = ResearchCoPilotOrchestrator()
