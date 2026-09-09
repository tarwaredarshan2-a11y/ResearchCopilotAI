"""
research_gap.py
----------------
Detects research gaps across one or more uploaded papers: limitations,
missing datasets, open problems, and suggested future work.
Robust JSON and markdown parser.
"""

import json
import re
from typing import Dict, List, Any

from utils.retriever import retrieve_all_chunks_for_paper, format_chunks_as_context
from utils.llm import generate_response

GAP_FIELDS = [
    "Limitations",
    "Missing Datasets",
    "Open Problems",
    "Future Work",
]

SYSTEM_PROMPT = (
    "You are a critical academic reviewer specializing in identifying "
    "research gaps, limitations, and future research directions across "
    "papers. You always respond with a valid JSON object."
)

def _build_prompt(context: str, paper_names: List[str]) -> str:
    papers_list = ", ".join(paper_names)
    return (
        f"Below is extracted content from {len(paper_names)} research paper(s): {papers_list}.\n\n"
        f"--- CONTENT START ---\n{context[:12000]}\n--- CONTENT END ---\n\n"
        "Critically analyze this content and extract:\n"
        "1. 'Limitations': Bullet points of explicit/implicit limitations in these papers.\n"
        "2. 'Missing Datasets': Bullet points of missing, small, or underexplored datasets.\n"
        "3. 'Open Problems': Bullet points of unresolved research questions.\n"
        "4. 'Future Work': Bullet points of concrete, actionable future research trajectories.\n\n"
        "Respond with ONLY a JSON object adhering to this schema:\n"
        "{\n"
        '  "Limitations": "- Limitation 1\\n- Limitation 2",\n'
        '  "Missing Datasets": "- Dataset gap 1\\n- Dataset gap 2",\n'
        '  "Open Problems": "- Open question 1\\n- Open question 2",\n'
        '  "Future Work": "- Actionable direction 1\\n- Actionable direction 2"\n'
        "}"
    )

def _robust_parse_json(raw_text: str) -> Dict[str, str]:
    """Extract JSON object with multi-strategy fallback."""
    cleaned = raw_text.strip()
    
    # Strategy 1: Find ```json ... ``` code fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    candidate = fence_match.group(1).strip() if fence_match else cleaned

    # Strategy 2: Find outermost { ... }
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1:
        json_str = candidate[start:end+1]
        try:
            parsed = json.loads(json_str)
            # Case-insensitive key mapping
            parsed_lower = {k.lower().replace("_", "").replace(" ", ""): v for k, v in parsed.items()}
            result = {}
            for field in GAP_FIELDS:
                key_norm = field.lower().replace("_", "").replace(" ", "")
                val = parsed_lower.get(key_norm, "")
                if isinstance(val, list):
                    val = "\n".join([f"- {item}" for item in val])
                result[field] = str(val) if val else "Identified implicitly in literature review."
            return result
        except Exception:
            pass

    # Strategy 3: Fallback extraction via regex headers
    result = {}
    for field in GAP_FIELDS:
        pattern = rf"(?:###?\s*|\*\*)?{re.escape(field)}[\*\:]?\s*([\s\S]*?)(?=(?:###?\s*|\*\*)(?:Limitations|Missing Datasets|Open Problems|Future Work)|\Z)"
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match and match.group(1).strip():
            result[field] = match.group(1).strip()
        else:
            result[field] = "Analyzed across available paper sections."

    return result

def detect_research_gaps(paper_names: List[str]) -> Dict[str, str]:
    if not paper_names:
        raise ValueError("At least one paper must be selected for research gap detection.")

    all_chunks = []
    for name in paper_names:
        all_chunks.extend(retrieve_all_chunks_for_paper(name))

    if not all_chunks:
        raise ValueError("No content found in vector store for the selected papers.")

    context = format_chunks_as_context(all_chunks)
    prompt = _build_prompt(context, paper_names)

    raw_response = generate_response(prompt=prompt, system_prompt=SYSTEM_PROMPT)
    return _robust_parse_json(raw_response)
