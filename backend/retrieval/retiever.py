from __future__ import annotations

from sentence_transformers import SentenceTransformer

from retrieval.reranker import Reranker
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_search import HybridSearch
from retrieval.query_profiles import get_query_profile
from retrieval.inspection import build_retrieval_trace


class Retriever:

    def __init__(self, vector_store, paper_text=None, equations=None, section_debug=None):
        self.vector_store = vector_store
        self.paper_text = paper_text or ""
        self.paper_equations = equations or []
        self.section_debug = section_debug or {}
        self.last_trace = None
        self.available_sections = sorted(
            {
                str(chunk.get("section", "")).lower()
                for chunk in vector_store.chunks
                if isinstance(chunk, dict) and chunk.get("section")
            }
        )

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.reranker = Reranker()
        self.bm25 = BM25Retriever(vector_store.chunks)
        self.hybrid = HybridSearch()

    def _retrieve_once(self, query, allowed_sections=None, equation_only=False, top_k=4):
        query_embedding = self.embedding_model.encode(query)

        dense_results = self.vector_store.search_with_scores(
            query_embedding,
            top_k=max(50, top_k * 4),
            allowed_sections=allowed_sections,
            equation_only=equation_only,
        )

        dense_chunks = [chunk for chunk, _, _ in dense_results]

        sparse_results = self.bm25.search(
            query,
            top_k=max(50, top_k * 4),
            allowed_sections=allowed_sections,
            equation_only=equation_only,
        )

        candidate_children = self.hybrid.fuse_scores(
            dense_chunks,
            sparse_results,
        )

        reranked_results = self.reranker.rerank_with_scores(
            query,
            candidate_children,
            top_k=max(20, top_k * 4),
        )

        ranked_children = [
            item["chunk"]
            for item in reranked_results
        ]

        parent_map = {}
        for child in ranked_children:
            parent_id = child["parent_id"]
            if parent_id not in parent_map:
                parent_map[parent_id] = child
            if len(parent_map) >= top_k:
                break

        selected_chunks = list(parent_map.values())
        context_parts = [
            (
                f"[{parent['section']} | chunk_id={parent['child_id']} | parent_id={parent['parent_id']}]\n\n"
                f"{parent['parent_text']}"
            )
            for parent in selected_chunks
        ]
        context = "\n\n".join(context_parts).strip()

        return {
            "dense_results": dense_results,
            "sparse_results": sparse_results,
            "reranked_results": reranked_results,
            "selected_chunks": selected_chunks,
            "context": context,
        }

    def _is_context_sufficient(self, bundle, min_chunk_count, min_context_length):
        return (
            len(bundle["selected_chunks"]) >= min_chunk_count
            and len(bundle["context"].strip()) >= min_context_length
        )

    def retrieve_context_bundle(
        self,
        query,
        top_k=4,
        section_filter=None,
        query_name=None
    ):
        profile = get_query_profile(
            query_name,
            section_filter=section_filter,
            available_sections=self.available_sections,
        )

        fallback_groups = profile.get("fallback_groups") or [[]]
        equation_only = profile.get("equation_only", False)
        min_chunk_count = profile.get("min_chunk_count", 2)
        min_context_length = profile.get("min_context_length", 500)
        fallback_invocations = []

        attempts = []
        for group in fallback_groups:
            attempts.append(
                {
                    "mode": "section",
                    "allowed_sections": group or None,
                }
            )

        attempts.append(
            {
                "mode": "global",
                "allowed_sections": None,
            }
        )

        best_bundle = None
        for attempt in attempts:
            allowed_sections = attempt["allowed_sections"]
            bundle = self._retrieve_once(
                query,
                allowed_sections=allowed_sections,
                equation_only=equation_only,
                top_k=top_k,
            )
            fallback_invocations.append(
                {
                    "mode": attempt["mode"],
                    "allowed_sections": allowed_sections or [],
                    "retrieved_chunk_ids": [chunk["child_id"] for chunk in bundle["selected_chunks"]],
                    "context_length": len(bundle["context"]),
                }
            )

            if bundle["selected_chunks"]:
                best_bundle = bundle

            if self._is_context_sufficient(bundle, min_chunk_count, min_context_length):
                trace = build_retrieval_trace(
                    query_name=query_name,
                    query_text=query,
                    section_filter=section_filter,
                    allowed_sections=allowed_sections or [],
                    fallback_mode=attempt["mode"],
                    fallback_invocations=fallback_invocations,
                    dense_results=bundle["dense_results"],
                    sparse_results=bundle["sparse_results"],
                    reranked_results=bundle["reranked_results"],
                    selected_chunks=bundle["selected_chunks"],
                    prompt_context=bundle["context"],
                    status="ok",
                )
                result = {
                    "query": query,
                    "query_name": query_name,
                    "context": bundle["context"],
                    "selected_chunks": bundle["selected_chunks"],
                    "retrieval_trace": trace,
                    "status": "ok",
                }
                self.last_trace = trace
                return result

        if best_bundle and best_bundle["selected_chunks"]:
            trace = build_retrieval_trace(
                query_name=query_name,
                query_text=query,
                section_filter=section_filter,
                allowed_sections=[],
                fallback_mode="best_effort",
                fallback_invocations=fallback_invocations,
                dense_results=best_bundle["dense_results"],
                sparse_results=best_bundle["sparse_results"],
                reranked_results=best_bundle["reranked_results"],
                selected_chunks=best_bundle["selected_chunks"],
                prompt_context=best_bundle["context"],
                status="insufficient_evidence",
            )
            result = {
                "query": query,
                "query_name": query_name,
                "context": best_bundle["context"],
                "selected_chunks": best_bundle["selected_chunks"],
                "retrieval_trace": trace,
                "status": "insufficient_evidence",
            }
            self.last_trace = trace
            return result

        trace = build_retrieval_trace(
            query_name=query_name,
            query_text=query,
            section_filter=section_filter,
            allowed_sections=[],
            fallback_mode="none",
            fallback_invocations=fallback_invocations,
            dense_results=[],
            sparse_results=[],
            reranked_results=[],
            selected_chunks=[],
            prompt_context="",
            status="insufficient_evidence",
        )
        result = {
            "query": query,
            "query_name": query_name,
            "context": "",
            "selected_chunks": [],
            "retrieval_trace": trace,
            "status": "insufficient_evidence",
        }
        self.last_trace = trace
        return result

    def retrieve_context(
        self,
        query,
        top_k=4,
        section_filter=None,
        query_name=None
    ):
        return self.retrieve_context_bundle(
            query=query,
            top_k=top_k,
            section_filter=section_filter,
            query_name=query_name,
        )["context"]
