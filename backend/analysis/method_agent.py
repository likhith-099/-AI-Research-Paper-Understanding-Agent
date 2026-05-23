from analysis.llm_client import generate_with_claude


def generate_method_explanation(retriever):
    context = retriever.retrieve_context(
        "method architecture model framework training pipeline approach algorithm",
        top_k=4
    )

    prompt = f"""
You are an AI research assistant.

Using the context below, explain the method used in the research paper.

Requirements:
- Write 5 to 6 lines
- Explain the model architecture or algorithm
- Describe how the system works step by step
- Avoid bullet points

Context:
{context}
"""

    return generate_with_claude(prompt, max_tokens=220)

