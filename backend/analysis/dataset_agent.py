from analysis.section_service import generate_evidence_based_section


def generate_dataset_explanation(retriever):
    return generate_evidence_based_section(
        retriever,
        section_name="dataset",
        query_text="dataset experiments benchmark training data evaluation dataset",
        max_tokens=500,
        prompt="""
Explain the datasets used in the research paper.

Requirements:
- Write 5 to 6 lines
- Describe datasets used for training and evaluation
- Mention benchmarks or data sources
- Explain why the dataset is important for the experiments
- Avoid bullet points
""".strip(),
    )

