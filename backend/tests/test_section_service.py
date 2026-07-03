from analysis import section_service


class DummyRetriever:
    def retrieve_context_bundle(self, query, top_k=4, query_name=None):
        return {
            "context": "Relevant fallback context from the whole paper. It supports the answer.",
            "selected_chunks": [
                {
                    "child_id": 7,
                    "parent_id": 2,
                    "section": "conclusion",
                    "child_text": "Relevant fallback context from the whole paper.",
                }
            ],
            "retrieval_trace": {
                "query": query_name,
                "query_text": query,
                "allowed_sections": [],
                "retrieved_chunk_ids": [7],
                "retrieved_sections": ["conclusion"],
            },
            "status": "insufficient_evidence",
        }


def test_generate_evidence_based_section_uses_fallback_context(monkeypatch):
    monkeypatch.setattr(
        section_service,
        "generate_with_groq",
        lambda *args, **kwargs: (
            "This paper suggests a broader conclusion about the open problem space.",
            {"model": "stub"},
        ),
    )
    monkeypatch.setattr(
        section_service,
        "validate_section_output",
        lambda section_name, text: (True, "ok"),
    )

    payload = section_service.generate_evidence_based_section(
        DummyRetriever(),
        section_name="research_gaps",
        query_text="open problems conclusion future work",
        prompt="Explain the research gaps.",
        max_tokens=128,
    )

    assert payload["answer"].startswith("This paper suggests")
    assert payload["status"] == "ok_fallback"
