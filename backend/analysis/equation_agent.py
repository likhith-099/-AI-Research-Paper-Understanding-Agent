import re
from analysis.llm_client import generate_with_claude


def extract_equations(context):
    """
    Detect possible equations or formula patterns
    """
    equation_pattern = r"([A-Za-z0-9]+\s*=\s*[^.]+)"
    matches = re.findall(equation_pattern, context)
    return matches[:3]


def generate_equation_explanation(retriever):
    context = retriever.retrieve_context(
        "loss function equation formula objective optimization model training",
        top_k=4
    )

    equations = extract_equations(context)

    prompt = f"""
You are an AI research assistant.

Below are equations extracted from a research paper along with surrounding context.

Explain the most important equation in simple language.

Requirements:
- Write 5 to 6 lines
- Explain what the equation represents
- Describe how it is used in the method
- Avoid mathematical complexity in explanation

Equations:
{equations}

Context:
{context}
"""

    return generate_with_claude(prompt, max_tokens=220)

