"""
hybrid_retriever.py
-------------------
Hybrid Retrieval & Re-ranking Engine for Research Paper Co-Pilot.

Combines:
1. Dense Semantic Vector Search (ChromaDB + BGE Embeddings)
2. Sparse Lexical Search (BM25)
3. Reciprocal Rank Fusion (RRF) & Score Normalization
"""

import math
import re
from typing import List, Dict, Any, Optional
from collections import Counter
from utils.embeddings import get_vector_store
from config import RETRIEVER_TOP_K, HYBRID_DENSE_WEIGHT, HYBRID_SPARSE_WEIGHT, RRF_K

class BM25Retriever:
    """Lightweight in-memory BM25 sparse keyword retriever."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.total_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, documents: List[Dict[str, Any]]):
        """Fit BM25 index on a list of document chunks (with 'text' and 'metadata')."""
        self.corpus = documents
        self.total_docs = len(documents)
        if self.total_docs == 0:
            return

        self.doc_lengths = []
        self.doc_freqs = Counter()

        tokenized_corpus = []
        for doc in documents:
            tokens = self._tokenize(doc.get("text", ""))
            self.doc_lengths.append(len(tokens))
            tokenized_corpus.append(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.doc_freqs[t] += 1

        self.avgdl = sum(self.doc_lengths) / max(1, self.total_docs)

        # Compute IDF
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log(1 + (self.total_docs - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 10, paper_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search corpus using BM25 scoring."""
        if not self.corpus:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc in enumerate(self.corpus):
            # Optional paper filter
            if paper_name and doc.get("metadata", {}).get("paper_name") != paper_name:
                continue

            tokens = self._tokenize(doc.get("text", ""))
            doc_len = self.doc_lengths[idx]
            token_counts = Counter(tokens)
            score = 0.0

            for q_tok in query_tokens:
                if q_tok in token_counts:
                    tf = token_counts[q_tok]
                    idf_val = self.idf.get(q_tok, 0.1)
                    denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(1, self.avgdl)))
                    score += idf_val * ((tf * (self.k1 + 1)) / max(1e-6, denom))

            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = []
        for idx, score in scores[:top_k]:
            doc = self.corpus[idx]
            top_results.append({
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
                "sparse_score": score
            })
        return top_results


class HybridRetriever:
    """Orchestrates Dense Semantic + Sparse BM25 + Reciprocal Rank Fusion (RRF)."""
    def __init__(self):
        self.bm25 = BM25Retriever()
        self._is_bm25_initialized = False

    def sync_bm25_from_vector_store(self):
        """Fetch all documents from ChromaDB and populate BM25 index."""
        try:
            vs = get_vector_store()
            collection = vs._collection
            data = collection.get(include=["documents", "metadatas"])
            docs = []
            if data and "documents" in data and data["documents"]:
                for text, meta in zip(data["documents"], data["metadatas"]):
                    docs.append({"text": text, "metadata": meta or {}})
            self.bm25.fit(docs)
            self._is_bm25_initialized = True
        except Exception as e:
            print(f"[HybridRetriever] Warning: Could not sync BM25 index: {e}")

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVER_TOP_K,
        paper_name: Optional[str] = None,
        section_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute Hybrid Search:
        1. Dense ChromaDB similarity search
        2. Sparse BM25 keyword search
        3. Reciprocal Rank Fusion (RRF)
        """
        # Ensure BM25 is updated
        if not self._is_bm25_initialized:
            self.sync_bm25_from_vector_store()

        # 1. Dense search
        dense_results = []
        try:
            vs = get_vector_store()
            filter_dict = {}
            if paper_name:
                filter_dict["paper_name"] = paper_name

            raw_dense = vs.similarity_search_with_relevance_scores(
                query,
                k=top_k * 2,
                filter=filter_dict if filter_dict else None
            )
            for doc, score in raw_dense:
                dense_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "dense_score": float(score)
                })
        except Exception as e:
            print(f"[HybridRetriever] Dense search error: {e}")

        # 2. Sparse search
        sparse_results = self.bm25.search(query, top_k=top_k * 2, paper_name=paper_name)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, Dict[str, Any]] = {}

        # Add Dense Ranks
        for rank, item in enumerate(dense_results):
            text_key = item["text"].strip()
            if text_key not in rrf_scores:
                rrf_scores[text_key] = {
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "rrf_score": 0.0,
                    "dense_score": item["dense_score"],
                    "sparse_score": 0.0
                }
            rrf_scores[text_key]["rrf_score"] += HYBRID_DENSE_WEIGHT / (RRF_K + rank + 1)

        # Add Sparse Ranks
        for rank, item in enumerate(sparse_results):
            text_key = item["text"].strip()
            if text_key not in rrf_scores:
                rrf_scores[text_key] = {
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "rrf_score": 0.0,
                    "dense_score": 0.0,
                    "sparse_score": item["sparse_score"]
                }
            else:
                rrf_scores[text_key]["sparse_score"] = item["sparse_score"]
            rrf_scores[text_key]["rrf_score"] += HYBRID_SPARSE_WEIGHT / (RRF_K + rank + 1)

        # Sort combined results
        combined = list(rrf_scores.values())
        combined.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Section filter if requested
        if section_filter:
            filtered = [
                item for item in combined 
                if section_filter.lower() in item.get("metadata", {}).get("section", "").lower()
            ]
            if filtered:
                combined = filtered

        return combined[:top_k]

# Global singleton instance
hybrid_retriever = HybridRetriever()
