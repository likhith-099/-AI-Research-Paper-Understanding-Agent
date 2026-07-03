from analysis.section_service import generate_evidence_based_section


def generate_implementation_ideas(retriever):
    return generate_evidence_based_section(
        retriever,
        section_name="implementation",
        query_text="training pipeline implementation framework model training hyperparameters experiment setup",
        max_tokens=550,
        prompt="""
Explain how someone could implement or reproduce the research paper.

Requirements:
- Write 5 to 6 lines
- Describe possible implementation steps
- Mention models, datasets, or training methods
- Keep explanation practical and clear
- Avoid bullet points
""".strip(),
    )

