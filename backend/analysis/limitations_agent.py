from analysis.section_service import generate_evidence_based_section


def generate_limitations(retriever):
    return generate_evidence_based_section(
        retriever,
        section_name="limitations",
        query_text="limitations drawback challenge weakness assumption discussion",
        max_tokens=500,
        prompt="""
Explain the limitations of the research paper.

Requirements:
- Write 5 to 6 lines
- Describe weaknesses or constraints of the approach
- Mention assumptions or practical challenges
- Avoid bullet points
""".strip(),
    )

