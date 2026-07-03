import fitz  # PyMuPDF
import re


def extract_text_from_pdf(pdf_path):
    """
    Extracts full text from a PDF research paper.
    Returns a cleaned text string.
    """

    doc = fitz.open(pdf_path)

    full_text = []

    for page in doc:
        text = page.get_text()
        if text:
            full_text.append(text)

    doc.close()

    combined_text = "\n".join(full_text)

    # Preserve paragraph and section breaks so downstream section detection works.
    cleaned_text = re.sub(r"[ \t]+", " ", combined_text)

    return cleaned_text.strip()
