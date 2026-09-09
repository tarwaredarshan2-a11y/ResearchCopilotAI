"""
embeddings.py
-------------
Manages the HuggingFace embedding model and the Chroma vector store.
Optimized for instant page load with lazy model initialization and error recovery.
"""

import os
from typing import List, Optional
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import EMBEDDING_MODEL_NAME, VECTOR_DB_DIR, CHROMA_COLLECTION_NAME, UPLOAD_DIR

@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load and cache the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vector_store() -> Chroma:
    """Get (or create) the persistent Chroma vector store instance."""
    embedding_model = get_embedding_model()
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=VECTOR_DB_DIR,
    )

def add_documents_to_vector_store(documents: List[Document]) -> int:
    """Embed and store Document chunks into ChromaDB."""
    if not documents:
        raise ValueError("No documents provided to add to vector store.")
    vs = get_vector_store()
    vs.add_documents(documents)
    return len(documents)

def delete_paper_from_vector_store(paper_name: str) -> None:
    """Remove paper chunks from ChromaDB."""
    try:
        vs = get_vector_store()
        vs.delete(where={"paper_name": paper_name})
    except Exception as e:
        print(f"[Embeddings] Delete warning: {e}")

def get_all_paper_names() -> List[str]:
    """
    Retrieve unique paper names. First checks uploads folder and ChromaDB
    safely without blocking the UI on heavy model downloads.
    """
    paper_names = set()
    
    # 1. Check uploads directory for files
    if os.path.exists(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            if f.lower().endswith(".pdf"):
                paper_names.add(f)

    # 2. Check ChromaDB collection metadata if available
    try:
        if os.path.exists(VECTOR_DB_DIR):
            vs = get_vector_store()
            collection = vs.get(include=["metadatas"])
            metadatas = collection.get("metadatas", []) or []
            for meta in metadatas:
                if meta and meta.get("paper_name"):
                    paper_names.add(meta.get("paper_name"))
    except Exception:
        pass

    return sorted(list(paper_names))
