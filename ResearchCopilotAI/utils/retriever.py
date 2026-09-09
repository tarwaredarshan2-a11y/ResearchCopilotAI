"""
retriever.py
------------
Provides retrieval functions that query the Chroma vector store for the
most relevant chunks to a given question, optionally scoped to a single
paper (for AI Chat / Paper Analysis) or across all papers (for
Literature Review / Research Gap Detection).
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
    """
    Retrieve the most relevant text chunks for a query from the vector
    store, optionally filtered to a single paper.

    Args:
        query (str): The natural-language question or search query.
        paper_name (Optional[str]): If provided, restricts retrieval to
            chunks whose metadata "paper_name" matches this value.
        top_k (int): Number of chunks to retrieve.

    Returns:
        List[Document]: The retrieved chunks, most relevant first.

    Raises:
        ValueError: If the query is empty.
    """
    if not query or not query.strip():
        raise ValueError("Query text cannot be empty.")

    vector_store = get_vector_store()

    search_kwargs = {"k": top_k}
    if paper_name:
        search_kwargs["filter"] = {"paper_name": paper_name}

    results = vector_store.similarity_search(query, **search_kwargs)
    return results


def retrieve_all_chunks_for_paper(paper_name: str) -> List[Document]:
    """
    Retrieve every stored chunk belonging to a specific paper, ordered
    by their original chunk_index. Useful for full-document tasks like
    Paper Analysis, where broad context (not just top-k similarity
    matches) is needed.

    Args:
        paper_name (str): The paper's identifying name.

    Returns:
        List[Document]: All chunks for the paper, ordered by chunk_index.
    """
    vector_store = get_vector_store()

    result = vector_store.get(
        where={"paper_name": paper_name},
        include=["documents", "metadatas"],
    )

    docs = result.get("documents", []) or []
    metas = result.get("metadatas", []) or []

    combined = list(zip(docs, metas))
    combined.sort(key=lambda item: item[1].get("chunk_index", 0))

    return [
        Document(page_content=doc, metadata=meta) for doc, meta in combined
    ]


def format_chunks_as_context(chunks: List[Document]) -> str:
    """
    Format a list of retrieved chunks into a single context string
    suitable for inserting into an LLM prompt.

    Args:
        chunks (List[Document]): Chunks to format.

    Returns:
        str: A newline-separated string of chunk contents, each labeled
            with its source paper.
    """
    if not chunks:
        return ""

    formatted_sections = []
    for i, chunk in enumerate(chunks, start=1):
        paper = chunk.metadata.get("paper_name", "Unknown Paper")
        formatted_sections.append(
            f"[Chunk {i} | Source: {paper}]\n{chunk.page_content}"
        )

    return "\n\n".join(formatted_sections)
