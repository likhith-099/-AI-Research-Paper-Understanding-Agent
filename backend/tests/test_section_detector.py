from tools.section_detector import detect_sections_verbose


def test_detects_ieee_and_survey_headings():
    text = """
I. INTRODUCTION
This paper introduces the problem.

II. RELATED WORK
Prior work is discussed here.

A. Taxonomy
We organize the literature by categories.

III. CONCLUSION AND FUTURE WORK
We summarize the findings and describe open problems.

IV. FUTURE WORK
Open directions are listed here.
""".strip()

    verbose = detect_sections_verbose(text)

    detected = {
        info["canonical_section"]
        for info in verbose["sections"].values()
    }

    assert "introduction" in detected
    assert "related work" in detected
    assert "future work" in detected
    assert "conclusion" in detected
    assert verbose["coverage"] >= 4 / 18


def test_detect_sections_falls_back_to_full_paper():
    text = "This paper has no explicit headings but still contains content."

    verbose = detect_sections_verbose(text)

    assert "full paper" in verbose["sections"]
    assert verbose["coverage"] == 0.0
