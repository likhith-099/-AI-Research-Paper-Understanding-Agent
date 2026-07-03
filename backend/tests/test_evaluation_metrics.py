from evaluation.metrics import build_ragas_compatible_report, faithfulness, answer_relevance


def test_build_ragas_report_uses_proxy_context_precision_without_labels():
    report = build_ragas_compatible_report(
        retrieved_ids=[1, 2, 3],
        relevant_ids=[],
        answer="The method improves robustness and scalability.",
        context="The method improves robustness and scalability.",
        query_text="method scalability robustness",
        retrieved_sections=["method", "method", "conclusion"],
        expected_sections=["method"],
    )

    assert report["mode"] == "proxy"
    assert report["retrieval"]["context_precision"] > 0
    assert report["generation"]["faithfulness"] >= 0
    assert report["generation"]["answer_relevance"] > 0


def test_faithfulness_and_relevance_respond_to_overlap():
    context = "The proposed method improves scalability on benchmark data."
    answer = "The proposed method improves scalability on benchmark data."

    assert faithfulness(answer, context) == 1.0
    assert answer_relevance(answer, "improves scalability benchmark") > 0.5
