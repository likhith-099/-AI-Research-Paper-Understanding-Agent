from analysis.section_service import generate_evidence_based_section


def generate_research_gaps(retriever):
    return generate_evidence_based_section(
        retriever,
        section_name="research_gaps",
        query_text="limitations drawback open problem future work conclusion future directions challenge scalability assumption unexplored area",
        max_tokens=650,
        prompt="""
Analyze the paper and identify potential research gaps.

Requirements:
- Identify 3 research gaps
- Each gap explanation should be about 5 to 6 lines
- Focus on limitations, missing experiments, scalability issues, unexplored areas, and what the conclusion or future work suggests remains open
- Avoid bullet points
""".strip(),
    )

