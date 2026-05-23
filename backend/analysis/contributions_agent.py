from analysis.llm_client import generate_with_claude


def generate_contributions(retriever):
    context = retriever.retrieve_context(
        "main contributions novelty proposed method improvement research contribution",
        top_k=4
    )

    prompt = f"""
You are an AI research assistant.

From the context below, explain the key contributions of the research paper.

Requirements:
- Write 3 to 4 lines
- Explain what is new in the paper
- Mention technical improvements or innovations
- Avoid bullet points

Context:
{context}
"""

    return generate_with_claude(prompt, max_tokens=220)

