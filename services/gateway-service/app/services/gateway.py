import time
from typing import List, Optional
import httpx
from app.core.config import get_settings
from app.schemas.gateway import (
    ChatContextSource,
    ChatExecutionTiming,
    ChatRequest,
    ChatResponse,
)

settings = get_settings()


class GatewayService:
    """Orchestration service coordinating calls between retrieval and generation services."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Orchestrates RAG flow:

        1. Calls retrieval-service for relevant passages.
        2. Calls generation-service to synthesize answer.
        3. Returns combined result to client.
        Falls back to mock data if downstream services are currently offline.
        """
        start_overall = time.perf_counter()
        retrieval_ms = 0.0
        generation_ms = 0.0

        retrieved_contexts: List[ChatContextSource] = []
        citations: List[str] = []
        answer = ""
        model_used = request.model or "gpt-4o-mini"

        # Step 1: Retrieval
        retrieval_start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.settings.REQUEST_TIMEOUT_SECONDS) as client:
                res = await client.post(
                    f"{self.settings.RETRIEVAL_SERVICE_URL}/api/v1/retrieve",
                    json={"query": request.query, "top_k": request.top_k or 5},
                )
                if res.status_code == 200:
                    data = res.json()
                    for item in data.get("results", []):
                        retrieved_contexts.append(
                            ChatContextSource(
                                chunk_id=item.get("chunk_id", "chk_mock"),
                                document_id=item.get("document_id", "doc_mock"),
                                content=item.get("content", ""),
                                score=item.get("score", 0.9),
                                metadata=item.get("metadata", {}),
                            )
                        )
                else:
                    retrieved_contexts = self._get_mock_contexts(request.query)
        except Exception:
            # Fallback mock contexts if retrieval-service is not running
            retrieved_contexts = self._get_mock_contexts(request.query)
        retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)

        # Step 2: Generation
        generation_start = time.perf_counter()
        contexts_payload = [
            {"id": c.chunk_id, "content": c.content, "source": c.metadata.get("title", c.document_id)}
            for c in retrieved_contexts
        ]
        try:
            async with httpx.AsyncClient(timeout=self.settings.REQUEST_TIMEOUT_SECONDS) as client:
                res = await client.post(
                    f"{self.settings.GENERATION_SERVICE_URL}/api/v1/generate",
                    json={
                        "query": request.query,
                        "contexts": contexts_payload,
                        "model": request.model,
                        "temperature": request.temperature,
                    },
                )
                if res.status_code == 200:
                    gen_data = res.json()
                    answer = gen_data.get("answer", "")
                    model_used = gen_data.get("model", model_used)
                    citations = gen_data.get("citations", [])
                else:
                    answer, citations = self._get_mock_answer(request.query, retrieved_contexts)
        except Exception:
            # Fallback mock generation if generation-service is not running
            answer, citations = self._get_mock_answer(request.query, retrieved_contexts)
        generation_ms = round((time.perf_counter() - generation_start) * 1000, 2)

        total_ms = round((time.perf_counter() - start_overall) * 1000, 2)

        return ChatResponse(
            query=request.query,
            answer=answer,
            model=model_used,
            contexts=retrieved_contexts if request.include_contexts else None,
            citations=citations,
            timing=ChatExecutionTiming(
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                total_ms=total_ms,
            ),
        )

    def _get_mock_contexts(self, query: str) -> List[ChatContextSource]:
        """Provides mock context chunks when downstream is unreachable."""
        return [
            ChatContextSource(
                chunk_id="chk_gateway_01",
                document_id="doc_scientometrics_2026",
                content=(
                    f"Literature survey for '{query}': Scientific publications on RAG architectures "
                    "demonstrate a 35% increase in citation velocity within applied AI journals."
                ),
                score=0.94,
                metadata={"title": "Scientific Journal Publication Trend Tracking", "year": 2026},
            ),
            ChatContextSource(
                chunk_id="chk_gateway_02",
                document_id="doc_microservices_2026",
                content=(
                    "Decoupled microservice architecture for RAG ensures fault isolation, "
                    "allowing independent scaling of embedding inference, vector database indices, and LLM workers."
                ),
                score=0.88,
                metadata={"title": "Scalable AI Microservices Systems", "year": 2026},
            ),
        ]

    def _get_mock_answer(self, query: str, contexts: List[ChatContextSource]):
        """Provides mock answer when downstream generation is unreachable."""
        sources = [c.metadata.get("title", c.document_id) for c in contexts]
        answer = (
            f"Theo tổng hợp từ hệ thống theo dõi xu hướng bài báo khoa học ({', '.join(set(sources))}):\n\n"
            f"Về câu hỏi '{query}', các công bố gần đây ghi nhận sự dịch chuyển mạnh mẽ sang kiến trúc "
            "Microservices cho hệ thống RAG, kết hợp với các kỹ thuật reranking đa tầng nhằm đảm bảo độ chính xác và khả năng mở rộng quy mô lớn."
        )
        return answer, list(set(sources))
