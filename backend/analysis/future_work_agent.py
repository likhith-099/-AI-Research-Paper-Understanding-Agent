from analysis.llm_client import generate_with_claude


def generate_future_work(retriever):
    context = retriever.retrieve_context(
        "future work improvement further research extension discussion conclusion",
        top_k=4
    )

    prompt = f"""
You are an AI research assistant.

Using the context below, explain the future work suggested by the research paper.

Requirements:
- Write 4 to 5 lines
- Describe possible improvements or extensions
- Mention potential research directions
- Avoid bullet points

Context:
{context}
"""

    return generate_with_claude(prompt, max_tokens=220)

