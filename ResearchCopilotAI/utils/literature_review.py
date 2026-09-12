"""
literature_review.py
---------------------
Generates a literature review across one or more uploaded papers:
an academic summary, a comparison table, research trends, strengths,
and weaknesses.
"""

import json
import re
from typing import Dict, List, Any

from utils.retriever import retrieve_all_chunks_for_paper, sample_balanced_chunks, format_chunks_as_context
from utils.llm import generate_response

REVIEW_FIELDS = [
    "Academic Summary",
    "Comparison Table",
    "Research Trends",
    "Strengths",
    "Weaknesses",
]

SYSTEM_PROMPT = (
    "You are an expert academic researcher who writes rigorous, well "
    "structured literature reviews comparing research papers. You always "
    "respond with a valid JSON object."
)

def _build_prompt(context: str, paper_names: List[str]) -> str:
    papers_list = ", ".join(paper_names)
    return (
        f"Below is extracted content from {len(paper_names)} research paper(s): {papers_list}.\n\n"
        f"--- CONTENT START ---\n{context[:14000]}\n--- CONTENT END ---\n\n"
        f"Generate a structured literature review specifically focusing on: {papers_list}.\n"
        "1. 'Academic Summary': Cohesive academic synthesis (3-4 paragraphs).\n"
        "2. 'Comparison Table': A markdown table comparing Paper Name, Methodology, Dataset, and Key Results.\n"
        "3. 'Research Trends': Bullet points describing common techniques and directions.\n"
        "4. 'Strengths': Bullet points of collective methodological strengths.\n"
        "5. 'Weaknesses': Bullet points of collective limitations.\n\n"
        "Respond with ONLY a JSON object adhering to this schema:\n"
        "{\n"
        '  "Academic Summary": "Synthesis text...",\n'
        '  "Comparison Table": "| Paper | Method | Dataset | Results |\\n|---|---|---|---|",\n'
        '  "Research Trends": "- Trend 1\\n- Trend 2",\n'
        '  "Strengths": "- Strength 1\\n- Strength 2",\n'
        '  "Weaknesses": "- Weakness 1\\n- Weakness 2"\n'
        "}"
    )

def _robust_parse_json(raw_text: str) -> Dict[str, str]:
    cleaned = raw_text.strip()
    
    # Strategy 1: Extract code fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    candidate = fence_match.group(1).strip() if fence_match else cleaned

    # Strategy 2: Extract { ... }
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1:
        json_str = candidate[start:end+1]
        try:
            parsed = json.loads(json_str)
            parsed_lower = {k.lower().replace("_", "").replace(" ", ""): v for k, v in parsed.items()}
            result = {}
            for field in REVIEW_FIELDS:
                key_norm = field.lower().replace("_", "").replace(" ", "")
                val = parsed_lower.get(key_norm, "")
                if isinstance(val, list):
                    val = "\n".join([f"- {item}" for item in val])
                result[field] = str(val) if val else "Synthesized from literature review."
            return result
        except Exception:
            pass

    # Strategy 3: Fallback extraction via regex
    result = {}
    for field in REVIEW_FIELDS:
        pattern = rf"(?:###?\s*|\*\*)?{re.escape(field)}[\*\:]?\s*([\s\S]*?)(?=(?:###?\s*|\*\*)(?:Academic Summary|Comparison Table|Research Trends|Strengths|Weaknesses)|\Z)"
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match and match.group(1).strip():
            result[field] = match.group(1).strip()
        else:
            result[field] = "Synthesized across analyzed papers."

    return result

def generate_literature_review(paper_names: List[str]) -> Dict[str, str]:
    if not paper_names:
        raise ValueError("At least one paper must be selected for literature review.")

    all_chunks = sample_balanced_chunks(paper_names, max_chunks_per_paper=8)

    if not all_chunks:
        raise ValueError("No content found in vector store for the selected papers.")

    context = format_chunks_as_context(all_chunks)
    prompt = _build_prompt(context, paper_names)

    raw_response = generate_response(prompt=prompt, system_prompt=SYSTEM_PROMPT)
    return _robust_parse_json(raw_response)
