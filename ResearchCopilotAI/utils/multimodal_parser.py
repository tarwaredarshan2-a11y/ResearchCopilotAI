"""
multimodal_parser.py
--------------------
Multimodal Ingestion & Academic Document Parser for Research Paper Co-Pilot.

Extracts structured text, page-level metadata, section boundaries, tables,
and figure references from academic PDFs with layout awareness.
"""

import os
import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional

SECTION_PATTERNS = [
    (r"^(?:abstract|summary)\b", "Abstract"),
    (r"^(?:1\.?|i\.?)\s*(?:introduction|overview)\b", "Introduction"),
    (r"^(?:2\.?|ii\.?)\s*(?:related\s*work|background|literature\s*review)\b", "Related Work"),
    (r"^(?:3\.?|iii\.?)\s*(?:methodology|methods|system\s*architecture|proposed\s*method)\b", "Methodology"),
    (r"^(?:4\.?|iv\.?)\s*(?:experiments|experimental\s*setup|evaluation|results)\b", "Experiments & Results"),
    (r"^(?:5\.?|v\.?)\s*(?:discussion|limitations|research\s*gaps)\b", "Discussion & Gaps"),
    (r"^(?:6\.?|vi\.?)\s*(?:conclusion|future\s*work|concluding\s*remarks)\b", "Conclusion"),
    (r"^(?:references|bibliography)\b", "References"),
]

def detect_section_header(line: str) -> Optional[str]:
    """Check if a line matches a common academic paper section header."""
    cleaned = line.strip().lower()
    for pattern, section_name in SECTION_PATTERNS:
        if re.search(pattern, cleaned):
            return section_name
    return None

def parse_multimodal_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Parse a research paper PDF and extract multimodal elements:
    - Text grouped by pages and section headers
    - Figure counts and table metadata
    - Structured document chunks with metadata
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    paper_name = os.path.basename(pdf_path)
    total_pages = len(doc)
    
    pages_data: List[Dict[str, Any]] = []
    full_text_list: List[str] = []
    total_images = 0
    tables_found = 0
    current_section = "Introduction"
    
    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_num = page_idx + 1
        text = page.get_text("text")
        full_text_list.append(text)
        
        # Image extraction count
        image_list = page.get_images(full=True)
        img_count = len(image_list)
        total_images += img_count
        
        # Line-by-line section header detection
        lines = text.split("\n")
        page_sections = []
        for line in lines[:15]:  # Check top lines of page
            detected = detect_section_header(line)
            if detected:
                current_section = detected
                page_sections.append(detected)
        
        # Table detection heuristics (e.g. lines with Table X: or tab-separated data)
        table_matches = re.findall(r"(?:Table\s+\d+[:\.]|TABLE\s+[I|V|X]+)", text)
        tables_found += len(table_matches)
        
        pages_data.append({
            "page_number": page_num,
            "text": text,
            "char_count": len(text),
            "image_count": img_count,
            "section": current_section,
            "table_references": table_matches
        })
        
    doc.close()
    
    combined_text = "\n\n".join(full_text_list).strip()
    
    return {
        "paper_name": paper_name,
        "total_pages": total_pages,
        "total_images": total_images,
        "total_tables": tables_found,
        "pages": pages_data,
        "full_text": combined_text,
        "is_scanned": len(combined_text) < (total_pages * 50)
    }

from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_parsed_document(parsed_doc: Dict[str, Any], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Split parsed document into chunks while preserving page number,
    paper name, and section metadata for fine-grained citation tracking.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks_with_metadata: List[Dict[str, Any]] = []
    paper_name = parsed_doc["paper_name"]
    
    for page in parsed_doc["pages"]:
        page_text = page["text"]
        page_num = page["page_number"]
        section = page["section"]
        
        if not page_text.strip():
            continue
            
        page_chunks = splitter.split_text(page_text)
        for idx, chunk_text in enumerate(page_chunks):
            if len(chunk_text.strip()) > 30:
                chunks_with_metadata.append({
                    "text": chunk_text.strip(),
                    "metadata": {
                        "paper_name": paper_name,
                        "page_number": page_num,
                        "section": section,
                        "chunk_id": f"{paper_name}_p{page_num}_c{idx}"
                    }
                })
                
    return chunks_with_metadata
