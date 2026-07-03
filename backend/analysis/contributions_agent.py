from analysis.section_service import generate_evidence_based_section


def generate_contributions(retriever):
    return generate_evidence_based_section(
        retriever,
        section_name="contributions",
        query_text="main contributions novelty proposed method improvement research contribution",
        max_tokens=450,
        prompt="""
From the evidence below, explain the key contributions of the research paper.

Requirements:
- Write 3 to 4 lines
- Explain what is new in the paper
- Mention technical improvements or innovations
- Avoid bullet points
""".strip(),
    )

