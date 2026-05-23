from analysis.llm_client import generate_with_claude


def generate_implementation_ideas(retriever):
    context = retriever.retrieve_context(
        "training pipeline implementation framework model training hyperparameters experiment setup",
        top_k=4
    )

    prompt = f"""
You are an AI research assistant.

Using the context below, explain how someone could implement or reproduce the research paper.

Requirements:
- Write 5 to 6 lines
- Describe possible implementation steps
- Mention models, datasets, or training methods
- Keep explanation practical and clear
- Avoid bullet points

Context:
{context}
"""

    return generate_with_claude(prompt, max_tokens=220)

