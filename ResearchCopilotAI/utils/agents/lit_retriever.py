"""
lit_retriever.py
----------------
Agent 2: Literature Retrieval Agent.

Executes multi-facet hybrid retrieval across uploaded papers using
both dense vector similarity and sparse BM25 keyword matching.
"""

from typing import Dict, Any, List, Optional
from utils.agents.base_agent import BaseAgent
from utils.hybrid_retriever import hybrid_retriever

class LiteratureRetrievalAgent(BaseAgent):
    def __init__(self):
        super().__init__("agent2_lit_retriever")

    def retrieve_literature(
        self,
        decomposition: Dict[str, Any],
        selected_paper: Optional[str] = None,
        top_k_per_query: int = 4
    ) -> Dict[str, Any]:
        """
        Executes multi-facet hybrid retrieval based on Agent 1 decomposition.
        """
        retrieval_queries = decomposition.get("retrieval_queries", [])
        if not retrieval_queries:
            retrieval_queries = [decomposition.get("original_query", "")]

        collected_chunks: Dict[str, Dict[str, Any]] = {}
        query_breakdown: List[Dict[str, Any]] = []

        for q in retrieval_queries:
            chunks = hybrid_retriever.retrieve(
                query=q,
                top_k=top_k_per_query,
                paper_name=selected_paper
            )
            query_breakdown.append({
                "sub_query": q,
                "retrieved_count": len(chunks),
                "top_chunks": chunks[:2]
            })
            for c in chunks:
                text_key = c["text"].strip()
                if text_key not in collected_chunks:
                    collected_chunks[text_key] = c
                else:
                    # Accumulate RRF score if fetched by multiple sub-queries
                    collected_chunks[text_key]["rrf_score"] += c.get("rrf_score", 0.0)

        # Sort aggregated chunks by combined score
        ranked_chunks = list(collected_chunks.values())
        ranked_chunks.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)

        return {
            "total_unique_chunks": len(ranked_chunks),
            "retrieved_chunks": ranked_chunks,
            "query_breakdown": query_breakdown,
            "papers_covered": list({c.get("metadata", {}).get("paper_name", "Unknown") for c in ranked_chunks})
        }
