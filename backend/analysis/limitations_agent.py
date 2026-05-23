from analysis.llm_client import generate_with_claude


def generate_limitations(retriever):
    context = retriever.retrieve_context(
        "limitations drawback challenge weakness assumption discussion",
        top_k=4
    )

    prompt = f"""
You are an AI research assistant.

Using the context below, explain the limitations of the research paper.

Requirements:
- Write 5 to 6 lines
- Describe weaknesses or constraints of the approach
- Mention assumptions or practical challenges
- Avoid bullet points

Context:
{context}
"""

    return generate_with_claude(prompt, max_tokens=220)

