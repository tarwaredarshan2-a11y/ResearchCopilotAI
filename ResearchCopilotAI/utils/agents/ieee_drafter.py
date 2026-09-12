"""
ieee_drafter.py
---------------
Agent 6: Drafting & IEEE Formatting Agent.

Drafts publication-ready IEEE conference-aligned paper sections,
bracketed citations [1], BibTeX entries, and full IEEEtran LaTeX source code.
"""

from typing import Dict, Any, List
from utils.agents.base_agent import BaseAgent
from config import IEEE_SECTIONS

class IEEEDraftingAgent(BaseAgent):
    def __init__(self):
        super().__init__("agent6_ieee_drafter")

    def draft_ieee_paper(
        self,
        topic: str,
        verified_evidence: List[Dict[str, Any]],
        verified_claims: List[Dict[str, Any]],
        research_gaps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesizes verified research components into full IEEE conference sections.
        """
        evidence_lines = []
        for i, e in enumerate(verified_evidence[:6]):
            src = e.get("source_paper", "Reference")
            stmt = e.get("statement", "")
            pg = e.get("page_number", "1")
            evidence_lines.append(f"[{i+1}] {src}: {stmt} (p.{pg})")
        evidence_summary = "\n".join(evidence_lines)

        claim_lines = []
        for c in verified_claims[:5]:
            c_text = c.get("claim_text", "")
            c_verdict = c.get("verdict", "ENTAILED")
            c_conf = c.get("confidence_score", 0.9)
            claim_lines.append(f"- {c_text} (Verdict: {c_verdict}, Conf: {c_conf:.2f})")
        claims_summary = "\n".join(claim_lines)

        gap_lines = []
        for g in research_gaps[:3]:
            g_title = g.get("title", "")
            g_desc = g.get("description", "")
            g_dir = g.get("actionable_research_direction", "")
            gap_lines.append(f"- {g_title}: {g_desc} (Direction: {g_dir})")
        gaps_summary = "\n".join(gap_lines)

        if not evidence_summary:
            evidence_summary = "General foundation in multimodal RAG and academic verification."

        prompt = (
            "You are the IEEE Drafting Agent in an academic Research Co-Pilot system.\n"
            "Your mission is to compose a rigorous, formal IEEE conference-style research paper draft based on the verified findings below.\n\n"
            f"Research Topic / System Focus:\n\"{topic}\"\n\n"
            f"Verified Source Literature & Evidence:\n{evidence_summary}\n\n"
            f"Validated Claims (NLI Entailed):\n{claims_summary}\n\n"
            f"Synthesized Research Gaps & Future Trajectories:\n{gaps_summary}\n\n"
            "Structure the draft following strict IEEE Conference formatting guidelines:\n"
            "1. Title\n"
            "2. Abstract (concise, problem-method-results summary)\n"
            "3. I. INTRODUCTION\n"
            "4. II. RELATED WORK\n"
            "5. III. METHODOLOGY & SYSTEM ARCHITECTURE\n"
            "6. IV. EXPERIMENTAL RESULTS & VERIFICATION\n"
            "7. V. DISCUSSION & OPEN GAPS\n"
            "8. VI. CONCLUSION\n"
            "9. REFERENCES (IEEE standard format [1], [2]...)\n"
            "10. BIBTEX ENTRIES\n\n"
            "Respond strictly with a VALID JSON object adhering to this schema:\n"
            "{\n"
            '  "paper_title": "Proposed Title of IEEE Paper",\n'
            '  "abstract": "Formal 150-250 word academic abstract...",\n'
            '  "keywords": ["Retrieval-Augmented Generation", "Multi-Agent Systems", "Claim Verification", "Multimodal Ingestion", "IEEE"],\n'
            '  "sections": {\n'
            '    "introduction": "Full text of Section I...",\n'
            '    "related_work": "Full text of Section II with [1], [2] citations...",\n'
            '    "methodology": "Full text of Section III describing equations and agent workflows...",\n'
            '    "experiments_and_results": "Full text of Section IV with evaluation metrics and ablation analysis...",\n'
            '    "discussion_and_gaps": "Full text of Section V discussing open challenges...",\n'
            '    "conclusion": "Full text of Section VI summarizing contributions..."\n'
            "  },\n"
            '  "ieee_references": [\n'
            '    "[1] A. Author and B. Researcher, Title of Paper, IEEE Trans. Knowl. Data Eng., vol. 35, no. 4, pp. 120-132, 2024.",\n'
            '    "[2] C. Scholar et al., Multimodal RAG for Academic Literature, in Proc. IEEE Conf. Artif. Intell., 2025, pp. 45-56."\n'
            "  ],\n"
            '  "bibtex_entries": "@article{author2024title, author = {Author, A.}, title = {Title of Paper}, year = {2024}}"\n'
            "}\n\n"
            "Return ONLY valid JSON."
        )

        response_text = self.call_llm(prompt)
        parsed = self.parse_json_response(response_text)

        if "sections" not in parsed:
            parsed = {
                "paper_title": f"A Verifiable Multi-Agent Framework for Scientific Literature Synthesis: {topic}",
                "abstract": f"This paper presents a six-agent human-in-the-loop research assistant framework for {topic}. By coupling layout-aware PDF extraction with hybrid dense-sparse retrieval and LLM-assisted claim verification, the system assists researchers in comparative literature analysis and preliminary IEEE manuscript drafting.",
                "keywords": ["Retrieval-Augmented Generation", "Multi-Agent Systems", "Claim Verification", "IEEE Guidelines"],
                "sections": {
                    "introduction": f"Comprehensive literature synthesis remains a critical task in computer science research. Addressing {topic}, this study presents an evidence-grounded multi-agent framework...",
                    "related_work": "Recent advances in dense vector retrieval and large language models have transformed document question answering [1]. However, ungrounded outputs in automated academic synthesis persist [2]...",
                    "methodology": "The proposed architecture incorporates a six-agent pipeline: Query Decomposition, Hybrid Literature Retrieval (Dense BGE + Sparse BM25), Citation Grounding, Claim Verification, Gap Synthesis, and IEEE Section Drafting...",
                    "experiments_and_results": "This draft contains preliminary literature synthesis. Experimental evaluation is planned using annotated research queries and evidence passages to measure precision, retrieval MRR, and citation grounding accuracy against primary source documents...",
                    "discussion_and_gaps": "While our multi-agent framework assists literature review, complex diagrammatic equation parsing in multi-column PDFs remains an open challenge requiring human verification...",
                    "conclusion": "This paper presented a multi-agent framework for evidence-grounded literature synthesis and preliminary IEEE conference draft generation."
                },
                "ieee_references": [
                    "[1] J. Devlin et al., 'BERT: Pre-training of Deep Bidirectional Transformers', in NAACL-HLT, 2019.",
                    "[2] P. Lewis et al., 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks', in NeurIPS, 2020."
                ],
                "bibtex_entries": "@inproceedings{lewis2020rag,\n  title={Retrieval-augmented generation for knowledge-intensive nlp tasks},\n  author={Lewis, Patrick and others},\n  booktitle={NeurIPS},\n  year={2020}\n}"
            }


        # Generate LaTeX Overleaf source string automatically
        sec = parsed.get("sections", {})
        title = parsed.get("paper_title", topic)
        abstract = parsed.get("abstract", "")
        kws = ", ".join(parsed.get("keywords", []))
        
        latex_str = f"""\\documentclass[conference]{{IEEEtran}}
\\usepackage{{cite}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{algorithmic}}
\\usepackage{{graphicx}}
\\usepackage{{textcomp}}
\\usepackage{{xcolor}}

\\begin{{document}}

\\title{{{title}}}

\\author{{\\IEEEauthorblockN{{Author Name}}
\\IEEEauthorblockA{{\\textit{{Department of Computer Science}} \\\\
\\textit{{University Name}}\\\\
City, Country \\\\
email@domain.com}}}}

\\maketitle

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\begin{{IEEEkeywords}}
{kws}
\\end{{IEEEkeywords}}

\\section{{Introduction}}
{sec.get('introduction', '')}

\\section{{Related Work}}
{sec.get('related_work', '')}

\\section{{Methodology \\& System Architecture}}
{sec.get('methodology', '')}

\\section{{Experimental Results \\& Verification}}
{sec.get('experiments_and_results', '')}

\\section{{Discussion \\& Research Gaps}}
{sec.get('discussion_and_gaps', '')}

\\section{{Conclusion}}
{sec.get('conclusion', '')}

\\begin{{thebibliography}}{{00}}
"""
        for idx, ref in enumerate(parsed.get("ieee_references", [])):
            clean_ref = ref.split("]", 1)[-1].strip() if "]" in ref else ref
            latex_str += f"\\bibitem{{ref{idx+1}}} {clean_ref}\n"

        latex_str += "\\end{thebibliography}\n\\end{document}"
        parsed["latex_source"] = latex_str

        return parsed
