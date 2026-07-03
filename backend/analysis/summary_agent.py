from analysis.section_service import generate_evidence_based_section


def generate_summary(retriever):
    return generate_evidence_based_section(
        retriever,
        section_name="summary",
        query_text="abstract summary main idea contribution of the paper",
        max_tokens=650,
        prompt="""
Using the evidence below, write a short summary of the research paper.

Requirements:
- 7 to 8 lines
- clear explanation
- explain the main idea and goal of the paper
- avoid bullet points
""".strip(),
    )

