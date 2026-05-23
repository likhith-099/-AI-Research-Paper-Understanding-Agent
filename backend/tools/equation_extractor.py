import re


def extract_equations(text):

    equation_pattern = r"[A-Za-z0-9]+\s*=\s*[^.]+"

    equations = re.findall(equation_pattern, text)

    return equations[:5]