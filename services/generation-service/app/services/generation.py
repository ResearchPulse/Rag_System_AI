import time
from app.core.config import get_settings
from app.schemas.generation import (
    GenerationRequest,
    GenerationResponse,
    GenerationUsage,
)

settings = get_settings()


class GenerationService:
    """Service handling prompt construction and LLM answer generation."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def generate_answer(self, request: GenerationRequest) -> GenerationResponse:
        """Constructs grounded answer using provided contexts and selected LLM."""
        # TODO: Integrate real LLM providers (LangChain / LlamaIndex / OpenAI / Anthropic / vLLM)
        start_time = time.perf_counter()
        model_name = request.model or self.settings.LLM_MODEL

        citations = []
        context_snippets = []
        for ctx in request.contexts:
            if ctx.source:
                citations.append(ctx.source)
            context_snippets.append(ctx.content[:100] + "...")

        # Mock synthesized grounded answer
        citations_text = ", ".join(set(citations)) if citations else "Internal scientific literature corpus"
        answer = (
            f"Dựa trên các tài liệu nghiên cứu được cung cấp ({citations_text}):\n\n"
            f"Đối với câu hỏi '{request.query}', các nghiên cứu mới nhất chỉ ra rằng xu hướng chủ đạo "
            "tập trung vào việc kết hợp mô hình RAG với kiến trúc microservices độc lập, tối ưu hóa quá trình chunking "
            "theo cấu trúc bài báo khoa học, và áp dụng reranking 2 tầng để nâng cao độ chính xác truy xuất."
        )

        prompt_tokens = sum(len(c.content.split()) for c in request.contexts) + len(request.query.split()) + 30
        completion_tokens = len(answer.split())
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return GenerationResponse(
            query=request.query,
            answer=answer,
            model=model_name,
            contexts_used=len(request.contexts),
            citations=list(set(citations)),
            usage=GenerationUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_ms=latency_ms,
        )
