from analysis.llm_client import generate_with_claude


def generate_summary(retriever):
    context = retriever.retrieve_context(
        "abstract summary main idea contribution of the paper",
        top_k=4
    )

    prompt = f"""
You are an AI research assistant.

Using the context below, write a short summary of the research paper.

Requirements:
- 7 to 8 lines
- clear explanation
- explain the main idea and goal of the paper
- avoid bullet points

Context:
{context}
"""

    return generate_with_claude(prompt, max_tokens=220)

