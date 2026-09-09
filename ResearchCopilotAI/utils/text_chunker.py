"""
text_chunker.py
----------------
Splits raw extracted PDF text into overlapping chunks suitable for
embedding and storage in the vector database. Uses LangChain's
RecursiveCharacterTextSplitter for semantically-aware splitting.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text_into_chunks(text: str, paper_name: str) -> List[Document]:
    """
    Split raw text into overlapping chunks and wrap each chunk as a
    LangChain Document with metadata identifying its source paper.

    Args:
        text (str): The full raw text extracted from a PDF.
        paper_name (str): The name of the paper this text belongs to.
            Stored in each chunk's metadata for later filtering/retrieval.

    Returns:
        List[Document]: A list of LangChain Document objects, each
            containing a chunk of text and metadata:
                {
                    "paper_name": str,
                    "chunk_index": int
                }

    Raises:
        ValueError: If the input text is empty.
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)

    documents: List[Document] = []
    for idx, chunk in enumerate(raw_chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "paper_name": paper_name,
                    "chunk_index": idx,
                },
            )
        )

    return documents
