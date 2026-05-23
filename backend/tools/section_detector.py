import re


def detect_sections(text):

    sections = {
        "abstract": "",
        "introduction": "",
        "method": "",
        "experiments": "",
        "conclusion": ""
    }

    patterns = {
        "abstract": r"abstract",
        "introduction": r"introduction",
        "method": r"(method|approach|model)",
        "experiments": r"(experiment|evaluation)",
        "conclusion": r"conclusion"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = match.start()
            sections[key] = text[start:start+2000]

    return sections