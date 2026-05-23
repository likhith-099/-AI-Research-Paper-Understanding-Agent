from analysis.llm_client import generate_with_claude


def generate_research_gaps(retriever):
    context = retriever.retrieve_context(
        "limitations drawback open problem future work challenge scalability assumption",
        top_k=4
    )

    prompt = f"""
You are an AI research assistant.

Analyze the context from the research paper and identify potential research gaps.

Requirements:
- Identify 3 research gaps
- Each gap explanation should be about 5 to 6 lines
- Focus on limitations, missing experiments, scalability issues, or unexplored areas
- Avoid bullet points

Context:
{context}
"""

    return generate_with_claude(prompt, max_tokens=220)

