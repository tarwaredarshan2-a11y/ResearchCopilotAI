"""
evaluation_suite.py
-------------------
Academic Benchmarking & Evaluation Suite for Research Paper Co-Pilot.

Implements quantitative evaluation metrics:
- Precision@K & MRR for retrieval accuracy
- Faithfulness / Grounding Score
- Hallucination Rate
"""

from typing import Dict, Any, List

class EvaluationSuite:
    """Calculates retrieval, verification, and grounding benchmarks."""

    @staticmethod
    def compute_retrieval_metrics(retrieved_docs: List[Dict[str, Any]], relevant_keywords: List[str], k: int = 5) -> Dict[str, float]:
        """Compute Precision@K and Reciprocal Rank."""
        if not retrieved_docs:
            return {"precision_at_k": 0.0, "mrr": 0.0}

        top_k_docs = retrieved_docs[:k]
        hits = 0
        first_hit_rank = 0

        for idx, doc in enumerate(top_k_docs):
            doc_text = doc.get("text", "").lower()
            is_relevant = any(kw.lower() in doc_text for kw in relevant_keywords)
            if is_relevant:
                hits += 1
                if first_hit_rank == 0:
                    first_hit_rank = idx + 1

        precision = hits / max(1, len(top_k_docs))
        mrr = (1.0 / first_hit_rank) if first_hit_rank > 0 else 0.0

        return {
            f"precision_at_{k}": round(precision, 4),
            "mrr": round(mrr, 4)
        }

    @staticmethod
    def compute_faithfulness_metrics(verified_claims: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute Faithfulness, Hallucination Rate, and Mean Entailment Confidence."""
        if not verified_claims:
            return {
                "faithfulness_score": 0.0,
                "hallucination_rate": 0.0,
                "average_confidence": 0.0,
                "total_claims": 0
            }

        total = len(verified_claims)
        entailed = sum(1 for c in verified_claims if c.get("verdict") == "ENTAILED")
        contradicted = sum(1 for c in verified_claims if c.get("verdict") == "CONTRADICTED" or c.get("is_hallucination"))
        avg_conf = sum(c.get("confidence_score", 0.0) for c in verified_claims) / total

        faithfulness = entailed / total
        hallucination = contradicted / total

        return {
            "faithfulness_score": round(faithfulness, 4),
            "hallucination_rate": round(hallucination, 4),
            "average_confidence": round(avg_conf, 4),
            "total_claims": total
        }

evaluation_suite = EvaluationSuite()
