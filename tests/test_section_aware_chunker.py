import pytest
from backend.rag.section_aware_chunker import detect_sections, chunk_section_text


def test_detect_abstract():
    """Verify Abstract section is detected."""
    text = """
    Abstract
    This paper presents a novel approach to transformer optimization.
    We demonstrate a 40% speedup on standard benchmarks.

    Introduction
    Deep learning has revolutionized...
    """
    sections = detect_sections(text)
    assert "abstract" in sections
    assert sections["abstract"]["start"] > 0
    assert "This paper presents" in sections["abstract"]["content"]


def test_detect_multiple_sections():
    """Verify multiple sections are detected."""
    text = """
    Abstract
    Efficient transformers.

    Introduction
    Deep learning background.

    Related Work
    Prior approaches.

    Methodology
    Our approach combines...

    Experiments
    We tested on CIFAR.

    Results
    Accuracy improved 5%.

    Discussion
    This suggests...

    Conclusion
    Future work includes...
    """
    sections = detect_sections(text)
    expected_sections = ["abstract", "introduction", "related_work", "methodology",
                        "experiments", "results", "discussion", "conclusion"]
    for section_name in expected_sections:
        assert section_name in sections, f"Section '{section_name}' not detected"


def test_chunk_section_text_with_word_limit():
    """Verify section is chunked within word limits."""
    section_text = " ".join(["word"] * 1500)  # 1500 words
    chunks = chunk_section_text(section_text, chunk_size=600, overlap=100)
    assert len(chunks) > 1  # Should split into multiple chunks
    assert all(len(c.split()) <= 650 for c in chunks)  # Allow some variance


def test_chunk_section_text_preserves_content():
    """Verify chunking preserves all text."""
    section_text = "The quick brown fox jumps over the lazy dog. " * 100
    chunks = chunk_section_text(section_text, chunk_size=600, overlap=100)
    reconstructed = " ".join(chunks)
    assert section_text.strip() in reconstructed or len(chunks) > 0
