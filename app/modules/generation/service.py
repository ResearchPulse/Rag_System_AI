import time
from typing import List
from app.core.config import get_settings
from app.modules.generation.schemas import (
    GenerationRequest,
    GenerationResponse,
    GenerationUsage,
    RagPipelineRequest,
    RagPipelineResponse,
)
from app.modules.retrieval.schemas import RetrievalRequest
from app.modules.retrieval.service import RetrievalService


class GenerationService:
    """Service orchestrating Phase 4: Generation & Evaluation (Steps 10, 11, 12)."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Synthesizes grounded answer using provided contexts."""
        start_time = time.perf_counter()
        model_name = request.model or self.settings.LLM_MODEL

        # If no contexts found in DB / Graph, return appropriate response
        if not request.contexts:
            lower_q = request.query.lower().strip(" ?,.!;:~")
            chitchat_tokens = [
                "chào", "hello", "hi", "hey", "alo", "bạn là ai", "who are you",
                "cảm ơn", "thanks", "tạm biệt", "bye", "bạn làm được gì", "giúp", "hướng dẫn",
            ]
            if any(tok in lower_q for tok in chitchat_tokens) or len(lower_q.split()) <= 3:
                answer = (
                    "Xin chào bạn! Tôi là **Trợ lý AI Nghiên cứu Khoa học** (Scientific Journal Trend Tracking Assistant).\n\n"
                    "Tôi có thể hỗ trợ bạn:\n"
                    "- 📚 **Tra cứu bài báo & tóm tắt nghiên cứu**: Phân tích nội dung các công bố khoa học từ kho dữ liệu PostgreSQL (pgvector).\n"
                    "- 🕸️ **Khám phá Đồ thị Tri thức (Knowledge Graph)**: Tra cứu thông tin tác giả, mạng lưới đồng tác giả, danh sách xuất bản và trích dẫn từ Neo4j.\n"
                    "- 📈 **Phân tích xu hướng học thuật**: Xu hướng theo chủ đề (Topic), tạp chí (Journal) và năm xuất bản.\n\n"
                    "Bạn muốn tìm hiểu thông tin hoặc nghiên cứu về chủ đề gì hôm nay?"
                )
            else:
                answer = (
                    f"Hệ thống không tìm thấy bất kỳ bài báo khoa học, tác giả hay thông tin liên quan nào "
                    f"phù hợp với câu hỏi '{request.query}' trong cơ sở dữ liệu."
                )
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return GenerationResponse(
                query=request.query,
                answer=answer,
                model=model_name,
                contexts_used=0,
                citations=[],
                usage=GenerationUsage(
                    prompt_tokens=len(request.query.split()),
                    completion_tokens=len(answer.split()),
                    total_tokens=len(request.query.split()) + len(answer.split()),
                ),
                latency_ms=latency_ms,
            )

        citations = [ctx.source for ctx in request.contexts if ctx.source]
        citations_unique = list(set(citations)) if citations else []

        answer = ""
        if self.settings.GEMINI_API_KEY:
            try:
                import json
                import urllib.request

                context_blocks = []
                for idx, ctx in enumerate(request.contexts, 1):
                    src = ctx.source or f"Nguồn {idx}"
                    context_blocks.append(f"[{idx}] Tiêu đề / Nguồn: {src}\nNội dung: {ctx.content[:1000]}")
                context_str = "\n\n".join(context_blocks)

                prompt = (
                    "Bạn là Trợ lý Nghiên cứu Khoa học (Scientific Journal AI Assistant).\n"
                    "Dựa vào các bài báo khoa học và dữ liệu trích xuất từ cơ sở dữ liệu dưới đây, "
                    "hãy trả lời câu hỏi của người dùng một cách chính xác, học thuật, có dẫn chứng rõ ràng.\n\n"
                    "QUY TẮC BẮT BUỘC:\n"
                    "- Tuyệt đối chỉ trả lời dựa trên dữ liệu được cung cấp dưới đây.\n"
                    "- Không tự bịa đặt tác giả, bài báo hay số liệu không có trong tài liệu.\n"
                    "- Nếu tài liệu không chứa đủ thông tin để trả lời, hãy thành thật nêu rõ rằng "
                    "cơ sở dữ liệu chưa có thông tin về vấn đề này.\n\n"
                    f"--- CÁC TÀI LIỆU TRÍCH XUẤT TỪ HỆ THỐNG ---\n{context_str}\n\n"
                    f"--- CÂU HỎI ---\n{request.query}\n\n"
                    "Hãy trả lời bằng tiếng Việt và liệt kê các nguồn tham khảo chính xác:"
                )

                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.settings.GEMINI_API_KEY}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": request.temperature or 0.2, "maxOutputTokens": 2048},
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                    answer = data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                answer = ""

        if not answer:
            facts = "\n".join(f"- {c.content}" for c in request.contexts[:3])
            answer = (
                f"Dựa trên dữ liệu ghi nhận từ hệ thống ResearchPulse ({', '.join(citations_unique) if citations_unique else 'Cơ sở dữ liệu'}):\n"
                f"{facts}"
            )

        prompt_tokens = sum(len(c.content.split()) for c in request.contexts) + len(request.query.split()) + 25
        completion_tokens = len(answer.split())
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return GenerationResponse(
            query=request.query,
            answer=answer,
            model=model_name,
            contexts_used=len(request.contexts),
            citations=citations_unique,
            usage=GenerationUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_ms=latency_ms,
        )

    def execute_rag_pipeline(
        self,
        request: RagPipelineRequest,
        retrieval_service: RetrievalService,
    ) -> RagPipelineResponse:
        """Executes full in-process End-to-End RAG Pipeline (Steps 7 through 12)."""
        start_overall = time.perf_counter()

        # Step 7-9: Retrieval (In-process call without HTTP overhead)
        t0 = time.perf_counter()
        retrieval_res = retrieval_service.retrieve(
            RetrievalRequest(query=request.query, top_k=request.top_k or 5)
        )
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Step 10-12: Context Assembly & Generation
        t1 = time.perf_counter()
        from app.modules.generation.schemas import ContextItem
        contexts_for_gen = [
            ContextItem(
                id=c.chunk_id,
                content=c.content,
                source=c.metadata.get("title", c.document_id),
            )
            for c in retrieval_res.results
        ]
        gen_res = self.generate(
            GenerationRequest(
                query=request.query,
                contexts=contexts_for_gen,
                model=request.model,
                temperature=request.temperature,
            )
        )
        generation_ms = round((time.perf_counter() - t1) * 1000, 2)
        total_ms = round((time.perf_counter() - start_overall) * 1000, 2)

        return RagPipelineResponse(
            query=request.query,
            answer=gen_res.answer,
            model=gen_res.model,
            contexts=[c.model_dump() for c in retrieval_res.results] if request.include_contexts else None,
            citations=gen_res.citations,
            latency_breakdown={
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
            },
        )
