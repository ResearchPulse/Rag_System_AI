import time
from app.core.config import get_settings
from app.modules.retrieval.schemas import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)


class RetrievalService:
    """Service orchestrating Phase 3: Retrieval (Steps 7, 8, 9)."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Executes query expansion, vector/hybrid search, and cross-encoder reranking."""
        start_time = time.perf_counter()

        # Mock retrieved literature passages
        mock_chunks = [
            RetrievedChunk(
                chunk_id="chk_pub_01",
                document_id="doc_scientometrics_2026",
                content=(
                    f"Publication trends analysis for '{request.query}': "
                    "Over 45% of literature in 2025-2026 focuses on modular RAG, agentic workflows, "
                    "and domain-adapted embeddings in scholarly publications."
                ),
                score=0.92,
                rerank_score=0.96 if request.rerank else None,
                metadata={"title": "State of Scientific AI Publications 2026", "year": 2026, "quartile": "Q1"},
            ),
            RetrievedChunk(
                chunk_id="chk_pub_02",
                document_id="doc_rag_benchmark_2026",
                content=(
                    "Comparative evaluation of modular monolithic vs microservices architectures for RAG: "
                    "Modular monoliths yield zero network latency overhead for intra-module pipeline operations."
                ),
                score=0.87,
                rerank_score=0.90 if request.rerank else None,
                metadata={"title": "Modular Monoliths in AI Retrieval Systems", "year": 2026, "quartile": "Q1"},
            ),
        ]

        filtered = [
            c for c in mock_chunks
            if (c.rerank_score or c.score) >= (request.score_threshold or 0.0)
        ][: request.top_k]

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return RetrievalResponse(
            query=request.query,
            total_found=len(filtered),
            results=filtered,
            latency_ms=latency_ms,
        )
