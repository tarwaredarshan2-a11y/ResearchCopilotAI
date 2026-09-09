"""
pdf_loader.py
-------------
Handles saving uploaded PDF files to disk and extracting raw text from
them using PyMuPDF (fitz).
"""

import os
import fitz  # PyMuPDF

from config import UPLOAD_DIR


def save_uploaded_pdf(uploaded_file) -> str:
    """
    Save a Streamlit UploadedFile object to the uploads/ directory.

    Args:
        uploaded_file: A Streamlit UploadedFile object (from st.file_uploader).

    Returns:
        str: The full path to the saved PDF file on disk.

    Raises:
        ValueError: If the uploaded file is empty or invalid.
    """
    if uploaded_file is None:
        raise ValueError("No file provided to save_uploaded_pdf.")

    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if os.path.getsize(file_path) == 0:
        raise ValueError(f"Uploaded file '{uploaded_file.name}' is empty.")

    return file_path


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text content from a PDF file using PyMuPDF.

    Args:
        file_path (str): Path to the PDF file on disk.

    Returns:
        str: The concatenated text extracted from every page of the PDF.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If no extractable text is found in the PDF.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")

    text_parts = []

    try:
        with fitz.open(file_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                page_text = page.get_text("text")
                if page_text:
                    text_parts.append(page_text)
    except Exception as exc:
        raise ValueError(f"Failed to extract text from PDF '{file_path}': {exc}")

    full_text = "\n".join(text_parts).strip()

    if not full_text:
        raise ValueError(
            f"No extractable text found in PDF '{file_path}'. "
            "The file may be scanned/image-based and requires OCR."
        )

    return full_text


def get_pdf_page_count(file_path: str) -> int:
    """
    Return the number of pages in a PDF file.

    Args:
        file_path (str): Path to the PDF file on disk.

    Returns:
        int: Number of pages in the PDF.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")

    with fitz.open(file_path) as doc:
        return doc.page_count
