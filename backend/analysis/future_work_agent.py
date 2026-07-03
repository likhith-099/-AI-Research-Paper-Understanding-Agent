from analysis.section_service import generate_evidence_based_section


def generate_future_work(retriever):
    return generate_evidence_based_section(
        retriever,
        section_name="future_work",
        query_text="future work improvement further research extension discussion conclusion future directions open problems",
        max_tokens=500,
        prompt="""
Explain the future work suggested by the research paper.

Requirements:
- Write 4 to 5 lines
- Describe possible improvements or extensions
- Mention potential research directions
- Avoid bullet points
""".strip(),
    )

