from tools.section_detector import detect_sections_verbose
from tools.equation_extractor import extract_equations


def split_words(
    text,
    size,
    overlap
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + size

        chunks.append(
            " ".join(words[start:end])
        )

        start += size - overlap

    return chunks


def chunk_text(text):

    verbose_sections = detect_sections_verbose(text)
    sections = verbose_sections["sections"]

    records = []

    parent_id = 0
    child_id = 0

    for section_name, section_info in sections.items():

        section_text = section_info["content"] if isinstance(section_info, dict) else section_info
        section_confidence = section_info.get("confidence", 0.0) if isinstance(section_info, dict) else 0.0
        section_heading = section_info.get("heading", section_name) if isinstance(section_info, dict) else section_name
        canonical_section = section_info.get("canonical_section", section_name) if isinstance(section_info, dict) else section_name

        parent_chunks = split_words(
            section_text,
            size=800,
            overlap=150
        )

        for parent_chunk in parent_chunks:

            current_parent_id = parent_id
            parent_equations = extract_equations(parent_chunk)

            child_chunks = split_words(
                parent_chunk,
                size=150,
                overlap=30
            )

            for child_chunk in child_chunks:
                child_equations = extract_equations(child_chunk)
                equation_matches = child_equations or parent_equations

                records.append(
                    {
                        "parent_id": current_parent_id,
                        "child_id": child_id,
                        "section": canonical_section.lower(),
                        "section_label": section_info.get("section_label", canonical_section) if isinstance(section_info, dict) else canonical_section,
                        "section_key": section_name,
                        "section_confidence": section_confidence,
                        "section_heading": section_heading,
                        "parent_text": parent_chunk,
                        "child_text": child_chunk,
                        "equations": equation_matches,
                        "has_equation": bool(equation_matches),
                    }
                )

                child_id += 1

            parent_id += 1

    return records
