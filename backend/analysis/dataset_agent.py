from analysis.llm_client import generate_with_claude


def generate_dataset_explanation(retriever):
    context = retriever.retrieve_context(
        "dataset experiments benchmark training data evaluation dataset",
        top_k=4
    )

    prompt = f"""
You are an AI research assistant.

Using the context below, explain the datasets used in the research paper.

Requirements:
- Write 5 to 6 lines
- Describe datasets used for training and evaluation
- Mention benchmarks or data sources
- Explain why the dataset is important for the experiments
- Avoid bullet points

Context:
{context}
"""

    return generate_with_claude(prompt, max_tokens=220)

