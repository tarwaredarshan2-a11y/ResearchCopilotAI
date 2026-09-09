"""
paper_analysis.py
------------------
Extracts structured analysis fields (title, authors, abstract, keywords,
methodology, dataset, model, results, conclusion, references) from a
single uploaded paper using retrieved context + Gemini 2.5 Flash.
"""

import json
import re
from typing import Dict, Any

from utils.retriever import retrieve_all_chunks_for_paper, format_chunks_as_context
from utils.llm import generate_response

ANALYSIS_FIELDS = [
    "Title",
    "Authors",
    "Abstract",
    "Keywords",
    "Methodology",
    "Dataset",
    "Model",
    "Results",
    "Conclusion",
    "References",
]

SYSTEM_PROMPT = (
    "You are an expert research paper analyst. You read academic paper "
    "content and extract structured information precisely and concisely. "
    "You always respond with a valid JSON object."
)

def _build_prompt(context: str) -> str:
    fields_list = ", ".join(ANALYSIS_FIELDS)
    return (
        "Below is the extracted text content of a research paper:\n\n"
        f"--- PAPER CONTENT START ---\n{context[:12000]}\n--- PAPER CONTENT END ---\n\n"
        f"Extract the following fields from the paper: {fields_list}.\n"
        "- Authors: comma-separated list of authors.\n"
        "- Keywords: comma-separated list of keywords.\n"
        "- Methodology, Dataset, Model, Results, Conclusion: concise summary.\n"
        "- References: note key references or reference style.\n\n"
        "Respond with ONLY a JSON object whose keys are exactly:\n"
        f"{json.dumps(ANALYSIS_FIELDS)}"
    )

def _robust_parse_json(raw_text: str) -> Dict[str, str]:
    cleaned = raw_text.strip()
    
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    candidate = fence_match.group(1).strip() if fence_match else cleaned

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1:
        json_str = candidate[start:end+1]
        try:
            parsed = json.loads(json_str)
            parsed_lower = {k.lower().replace("_", "").replace(" ", ""): v for k, v in parsed.items()}
            result = {}
            for field in ANALYSIS_FIELDS:
                key_norm = field.lower().replace("_", "").replace(" ", "")
                val = parsed_lower.get(key_norm, "")
                if isinstance(val, list):
                    val = ", ".join(val)
                result[field] = str(val) if val else "Not clearly stated in the paper."
            return result
        except Exception:
            pass

    result = {}
    for field in ANALYSIS_FIELDS:
        pattern = rf"(?:###?\s*|\*\*)?{re.escape(field)}[\*\:]?\s*([\s\S]*?)(?=(?:###?\s*|\*\*)(?:Title|Authors|Abstract|Keywords|Methodology|Dataset|Model|Results|Conclusion|References)|\Z)"
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match and match.group(1).strip():
            result[field] = match.group(1).strip()
        else:
            result[field] = "Not clearly stated in the paper."

    return result

def analyze_paper(paper_name: str) -> Dict[str, str]:
    chunks = retrieve_all_chunks_for_paper(paper_name)
    if not chunks:
        raise ValueError(f"No content found in vector store for paper '{paper_name}'.")

    context = format_chunks_as_context(chunks)
    prompt = _build_prompt(context)

    raw_response = generate_response(prompt=prompt, system_prompt=SYSTEM_PROMPT)
    return _robust_parse_json(raw_response)
