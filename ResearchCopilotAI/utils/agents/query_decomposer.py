"""
query_decomposer.py
-------------------
Agent 1: Query Decomposition Agent.

Deconstructs complex academic queries into sub-questions, key search facets,
methodology constraints, and targeted literature retrieval queries.
"""

from typing import Dict, Any, List
from utils.agents.base_agent import BaseAgent

class QueryDecompositionAgent(BaseAgent):
    def __init__(self):
        super().__init__("agent1_query_decomposer")

    def decompose(self, user_query: str) -> Dict[str, Any]:
        """
        Deconstructs a user query into structured research sub-questions and search facets.
        """
        prompt = f"""You are the Query Decomposition Agent in an academic Research Paper Co-Pilot system.
Your task is to analyze the research query below and deconstruct it into structured research components.

User Research Query:
\"\"\"{user_query}\"\"\"

Respond with a VALID JSON object adhering strictly to this format:
{{
  "original_query": "{user_query}",
  "core_research_intent": "Clear 1-sentence statement of the main scientific question",
  "sub_questions": [
    "Specific sub-question 1 (e.g. architectural/methodological aspects)",
    "Specific sub-question 2 (e.g. datasets and experimental setup)",
    "Specific sub-question 3 (e.g. empirical results, ablation, or limitations)"
  ],
  "search_facets": [
    "keyword facet 1",
    "keyword facet 2",
    "keyword facet 3"
  ],
  "targeted_sections": ["Methodology", "Experiments & Results", "Discussion & Gaps"],
  "retrieval_queries": [
    "Dense/Sparse Search Query 1",
    "Dense/Sparse Search Query 2",
    "Dense/Sparse Search Query 3"
  ]
}}

Return ONLY valid JSON.
"""
        response_text = self.call_llm(prompt)
        parsed = self.parse_json_response(response_text)
        
        # Fallback structure if JSON parse fails
        if "sub_questions" not in parsed:
            parsed = {
                "original_query": user_query,
                "core_research_intent": user_query,
                "sub_questions": [user_query],
                "search_facets": [user_query],
                "targeted_sections": ["Methodology", "Experiments & Results"],
                "retrieval_queries": [user_query]
            }
        return parsed
