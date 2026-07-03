from analysis.section_service import generate_evidence_based_section


def generate_equation_explanation(retriever):
    equations = getattr(retriever, "paper_equations", [])[:8]

    if not equations:
        return {
            "answer": "No explicit mathematical equations found in paper.",
            "evidence_chunks": [],
            "source_sections": [],
            "status": "insufficient_evidence",
            "retrieval_debug": None,
            "generation_debug": None,
        }

    return generate_evidence_based_section(
        retriever,
        section_name="equations",
        query_text="loss function equation formula objective optimization model training",
        max_tokens=450,
        prompt=f"""
Below are equations extracted from a research paper.

Explain the most important equation in simple language.

Requirements:
- Write 5 to 6 lines
- Explain what the equation represents
- Describe how it is used in the method
- Avoid mathematical complexity in explanation

Equations:
{equations}
""".strip(),
    )

