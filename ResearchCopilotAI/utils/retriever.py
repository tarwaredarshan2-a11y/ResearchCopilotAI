"""
retriever.py
------------
Provides retrieval functions querying the Chroma vector store,
correctly sorted by page number and chunk sequence.
"""

from typing import List, Optional
from langchain_core.documents import Document

from config import RETRIEVER_TOP_K
from utils.embeddings import get_vector_store

def retrieve_relevant_chunks(
    query: str,
    paper_name: Optional[str] = None,
    top_k: int = RETRIEVER_TOP_K,
) -> List[Document]:
    if not query or not query.strip():
        raise ValueError("Query text cannot be empty.")

    vector_store = get_vector_store()
    search_kwargs = {"k": top_k}
    if paper_name:
        search_kwargs["filter"] = {"paper_name": paper_name}

    results = vector_store.similarity_search(query, **search_kwargs)
    return results

def retrieve_all_chunks_for_paper(paper_name: str) -> List[Document]:
    """Retrieve every chunk for a paper, correctly ordered by page_number."""
    vector_store = get_vector_store()
    result = vector_store.get(
        where={"paper_name": paper_name},
        include=["documents", "metadatas"],
    )

    docs = result.get("documents", []) or []
    metas = result.get("metadatas", []) or []

    combined = list(zip(docs, metas))
    combined.sort(key=lambda item: (item[1].get("page_number", 0), item[1].get("chunk_id", "")))

    return [
        Document(page_content=doc, metadata=meta) for doc, meta in combined
    ]

def sample_balanced_chunks(paper_names: List[str], max_chunks_per_paper: int = 10) -> List[Document]:
    """
    Retrieve an evenly distributed sample of chunks from each paper across
    beginning (Abstract/Intro), middle (Methodology), and end (Results/Conclusion).
    Prevents a single paper from dominating the context window.
    """
    sampled: List[Document] = []
    for name in paper_names:
        chunks = retrieve_all_chunks_for_paper(name)
        if not chunks:
            continue
        if len(chunks) <= max_chunks_per_paper:
            sampled.extend(chunks)
        else:
            # Sample evenly across the document
            step = len(chunks) / float(max_chunks_per_paper)
            indices = [int(i * step) for i in range(max_chunks_per_paper)]
            for idx in indices:
                if idx < len(chunks):
                    sampled.append(chunks[idx])
    return sampled

def format_chunks_as_context(chunks: List[Document]) -> str:
    if not chunks:
        return ""

    formatted_sections = []
    for i, chunk in enumerate(chunks, start=1):
        paper = chunk.metadata.get("paper_name", "Unknown Paper")
        page = chunk.metadata.get("page_number", "?")
        formatted_sections.append(
            f"[Chunk {i} | Paper: {paper} | Page: {page}]\n{chunk.page_content}"
        )

    return "\n\n".join(formatted_sections)

