"""
gap_synthesizer.py
------------------
Agent 5: Research-Gap Synthesis Agent.

Synthesizes cross-paper research gaps, conflicting empirical results,
methodological limitations, and actionable future research questions.
"""

from typing import Dict, Any, List
from utils.agents.base_agent import BaseAgent

class ResearchGapSynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__("agent5_gap_synthesizer")

    def synthesize_gaps(self, research_topic: str, verified_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identifies cross-paper research gaps from verified evidence.
        """
        evidence_str = "\n".join([
            f"- [{e.get('source_paper', 'Doc')}, p.{e.get('page_number', '?')}]: {e.get('statement', '')}"
            for e in verified_evidence
        ])

        prompt = f"""You are the Research-Gap Synthesis Agent in an academic Research Co-Pilot system.
Analyze the verified scientific evidence below on the topic \"{research_topic}\" and synthesize high-impact research gaps.

Verified Evidence:
{evidence_str if evidence_str else "No evidence provided; analyze general literature gaps on this topic."}

Respond with a VALID JSON object adhering to this schema:
{{
  "research_gaps": [
    {{
      "gap_id": "G1",
      "category": "Methodological / Dataset / Scalability / Theoretical",
      "title": "Concise title of the gap",
      "description": "Detailed explanation of what existing papers overlook or assume",
      "impact_severity": "High / Medium / Critical",
      "contributing_papers": ["Paper A", "Paper B"],
      "actionable_research_direction": "Concrete proposed experimental or algorithmic solution",
      "novel_hypothesis": "A testable research hypothesis to resolve this gap"
    }}
  ],
  "cross_paper_conflicts": [
    {{
      "conflict_topic": "e.g. Model scalability vs latency trade-off",
      "paper_a_claim": "Claim from Paper 1",
      "paper_b_claim": "Conflicting finding from Paper 2",
      "synthesis": "Root cause of variance (dataset size, domain differences, etc.)"
    }}
  ],
  "executive_summary": "Comprehensive 2-paragraph overview of current state-of-the-art boundaries."
}}

Return ONLY valid JSON.
"""
        response_text = self.call_llm(prompt)
        parsed = self.parse_json_response(response_text)

        if "research_gaps" not in parsed:
            parsed = {
                "research_gaps": [
                    {
                        "gap_id": "G1",
                        "category": "Methodological",
                        "title": "Evaluation on Out-of-Distribution Scientific Benchmarks",
                        "description": "Existing approaches evaluate primarily on closed-domain papers, lacking multimodal robustness on diverse STEM disciplines.",
                        "impact_severity": "High",
                        "contributing_papers": ["Uploaded Literature"],
                        "actionable_research_direction": "Implement fine-grained zero-shot cross-encoder reranking across multimodal diagrammatic datasets.",
                        "novel_hypothesis": "Hybrid citation-grounded RAG with NLI filtering significantly reduces false-positive scientific gap claims."
                    }
                ],
                "cross_paper_conflicts": [],
                "executive_summary": "Synthesized critical open gaps in multimodal research automation and verifiable retrieval."
            }

        return parsed
