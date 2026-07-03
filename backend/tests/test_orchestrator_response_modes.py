from agent import orchestrator


def test_analyze_paper_returns_public_payload_by_default(monkeypatch):
    def _build_payload(answer):
        return {
            "answer": answer,
            "retrieval_debug": {
                "query": "summary",
                "query_text": "summary",
                "retrieved_ids": [1],
                "retrieved_chunk_ids": [1],
                "retrieved_sections": ["abstract"],
                "context_length": 12,
                "faiss_scores": [0.9],
                "bm25_scores": [0.7],
                "rerank_scores": [0.8],
                "final_context_length": 12,
                "status": "ok",
                "fallback_mode": "section",
                "chunks_sent_to_groq": [{"child_id": 1}],
                "allowed_sections": ["abstract"],
            },
            "status": "ok",
        }

    monkeypatch.setattr(
        orchestrator,
        "SECTION_GENERATORS",
        [
            ("summary", lambda retriever: _build_payload("short summary")),
            ("contributions", lambda retriever: _build_payload("short contributions")),
        ],
    )
    monkeypatch.setattr(orchestrator, "PUBLIC_SECTION_ORDER", ["summary", "contributions"])

    result = orchestrator.analyze_paper(object(), include_debug=False)

    assert result == {
        "summary": "short summary",
        "contributions": "short contributions",
    }


def test_analyze_paper_returns_debug_payload_when_enabled(monkeypatch):
    def _build_payload(answer):
        return {
            "answer": answer,
            "retrieval_debug": {
                "query": "summary",
                "query_text": "summary",
                "retrieved_ids": [1],
                "retrieved_chunk_ids": [1],
                "retrieved_sections": ["abstract"],
                "context_length": 12,
                "faiss_scores": [0.9],
                "bm25_scores": [0.7],
                "rerank_scores": [0.8],
                "final_context_length": 12,
                "status": "ok",
                "fallback_mode": "section",
                "chunks_sent_to_groq": [{"child_id": 1}],
                "allowed_sections": ["abstract"],
            },
            "status": "ok",
        }

    monkeypatch.setattr(
        orchestrator,
        "SECTION_GENERATORS",
        [("summary", lambda retriever: _build_payload("short summary"))],
    )
    monkeypatch.setattr(orchestrator, "PUBLIC_SECTION_ORDER", ["summary"])

    result = orchestrator.analyze_paper(object(), include_debug=True)

    assert "debug_report" in result
    assert result["summary"]["answer"] == "short summary"
    assert result["debug_report"]["section_traces"]["summary"]["retrieved_ids"] == [1]
