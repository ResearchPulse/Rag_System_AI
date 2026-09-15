import time
from typing import Any, List, Optional
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
from app.modules.retrieval.query_rewriting.compressor import QueryCompressor


class GenerationService:
    """Service orchestrating Phase 4: Generation & Evaluation (Steps 10, 11, 12)."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Synthesizes grounded answer using provided contexts."""
        start_time = time.perf_counter()
        model_name = request.model or self.settings.LLM_MODEL
        if (not request.model or request.model in ["gpt-4o-mini", "mock", "gemini-1.5-flash"]) and self.settings.LLM_PROVIDER == "ollama":
            model_name = self.settings.OLLAMA_MODEL or "llama3.2:3b"

        # If no contexts found in DB / Graph, return appropriate response
        if not request.contexts:
            import re
            lower_q = request.query.lower().strip(" ?,.!;:~")
            is_chitchat = bool(re.search(
                r"\b(xin\s*chào|chào|hello|hi|hey|alo|bạn\s*là\s*ai|who\s*are\s*you|cảm\s*ơn|thanks|tạm\s*biệt|bye|hướng\s*dẫn)\b",
                lower_q,
                re.IGNORECASE,
            ))
            has_academic = any(k in lower_q for k in ["bài báo", "tác giả", "tác giác", "nghiên cứu", "paper", "author", "bao nhiêu", "tổng", "đếm", "thống kê"])
            if is_chitchat and not has_academic:
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
        compressed_q = QueryCompressor.compress_query(request.query)

        # Compact context blocks: top 4 contexts
        context_blocks = []
        for idx, ctx in enumerate(request.contexts[:4], 1):
            src = ctx.source or f"Nguồn {idx}"
            is_stat_agg = (
                isinstance(ctx.metadata, dict)
                and (ctx.metadata.get("type") == "Aggregation" or ctx.metadata.get("source") == "postgresql_sql")
            )
            if is_stat_agg:
                # Keep structured statistical tables & rankings intact without flattening newlines
                compact_content = ctx.content[:1500].strip()
            else:
                compact_content = QueryCompressor.compress_context(ctx.content, max_chars=350)
            context_blocks.append(f"[{idx}] {src}:\n{compact_content}")
        context_str = "\n\n".join(context_blocks)
        history_str = f"\n\n{request.history_context}\n" if getattr(request, "history_context", None) else ""

        is_ollama = (self.settings.LLM_PROVIDER == "ollama" or getattr(self.settings, "OLLAMA_BASE_URL", None))
        if is_ollama:
            prompt = (
                "Bạn là trợ lý AI học thuật ResearchPulse. Dựa vào TÀI LIỆU dưới đây, hãy trả lời súc tích, chính xác bằng tiếng Việt.\n"
                "QUY TẮC QUAN TRỌNG: Nếu trong TÀI LIỆU có bảng hoặc con số thống kê, BẮT BUỘC dùng đúng con số đó, tuyệt đối không tự bịa số liệu.\n\n"
                f"--- TÀI LIỆU ---\n{context_str}\n"
                f"{history_str}\n"
                f"--- CÂU HỎI ---\n{compressed_q}\n\n"
                "Trả lời:"
            )
            max_predict = min(self.settings.LLM_MAX_TOKENS or 512, 512)
        else:
            prompt = (
                "Bạn là Trợ lý Nghiên cứu Khoa học (Scientific Journal AI Assistant).\n"
                "Dựa vào các bài báo khoa học và dữ liệu trích xuất từ cơ sở dữ liệu dưới đây, "
                "hãy trả lời câu hỏi của người dùng một cách chính xác, học thuật, có dẫn chứng rõ ràng bằng tiếng Việt.\n\n"
                "QUY TẮC BẮT BUỘC:\n"
                "- Trả lời trung thực, học thuật, mạch lạc dựa trên các tài liệu được cung cấp dưới đây.\n"
                "- Nếu tài liệu là số liệu thống kê [Thống kê cơ sở dữ liệu ResearchPulse], hãy nêu rõ các con số chính xác.\n"
                "- Nêu rõ tên bài báo, năm xuất bản và tóm tắt các điểm then chốt.\n\n"
                f"--- CÁC TÀI LIỆU TRÍCH XUẤT TỪ HỆ THỐNG ---\n{context_str}\n"
                f"{history_str}\n"
                f"--- CÂU HỎI CỦA NGƯỜI DÙNG ---\n{compressed_q}\n\n"
                "Câu trả lời (bằng tiếng Việt):"
            )
            max_predict = self.settings.LLM_MAX_TOKENS or 1024


        # 1. Local Ollama LLM (e.g. llama3.2:3b)
        if is_ollama:
            try:
                import json
                import urllib.request
                ollama_url = f"{self.settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature or self.settings.LLM_TEMPERATURE or 0.3,
                        "num_predict": max_predict,
                    }
                }
                req = urllib.request.Request(
                    ollama_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    answer = data.get("response", "").strip()
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Ollama local generation error: {e}")

        # 2. Cloud Fallback (Gemini) only if LLM_PROVIDER is not ollama and key exists
        if not answer and self.settings.GEMINI_API_KEY and self.settings.LLM_PROVIDER != "ollama":
            try:
                import json
                import urllib.request
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
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                    answer = data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                pass

        if not answer:
            # 1. Statistical Aggregations (from PostgreSQL or Neo4j)
            sql_contexts = [
                c for c in request.contexts
                if c.metadata.get("type") == "Aggregation"
                or c.metadata.get("source") in ["postgresql_sql", "neo4j_graph"] and c.metadata.get("type") == "Aggregation"
                or "Thống kê cơ sở dữ liệu" in c.content
            ]
            if sql_contexts:
                answer = sql_contexts[0].content
            else:
                # 2. Relational Author / Knowledge Graph Contexts
                graph_contexts = [
                    c for c in request.contexts
                    if c.metadata.get("source") == "neo4j" or "Knowledge Graph" in c.content
                ]
                article_contexts = [
                    c for c in request.contexts
                    if c not in graph_contexts and c not in sql_contexts
                ]

                lower_q = request.query.lower()
                is_trend = any(k in lower_q for k in ["xu hướng", "hướng", "tiềm năng", "tương lai", "phát triển", "trend", "evolution"])
                is_author = any(k in lower_q for k in ["tác giả", "author", "hợp tác", "đồng tác giả", "cùng viết"])

                sections = []
                if is_trend:
                    sections.append(f"### 📈 Phân tích Xu hướng Nghiên cứu Khoa học\n**Chủ đề**: *{request.query}*\n\nDựa trên các tài liệu công bố khoa học mới nhất được ghi nhận trong cơ sở dữ liệu hệ thống ResearchPulse, xu hướng nghiên cứu và các công bố tiêu biểu bao gồm:")
                elif is_author and graph_contexts:
                    sections.append(f"### 👤 Thông tin Tác giả & Mạng lưới Học thuật trên Knowledge Graph\n**Truy vấn**: *{request.query}*\n")
                else:
                    sections.append(f"### 📚 Kết quả Tra cứu & Tổng hợp Khoa học\n**Truy vấn**: *{request.query}*\n\nDựa trên dữ liệu ghi nhận từ hệ thống ResearchPulse:")

                # Render Graph contexts if available
                if graph_contexts:
                    for gc in graph_contexts:
                        sections.append(gc.content)

                # Render Article contexts
                if article_contexts:
                    sections.append("#### 📑 Các công bố khoa học tiêu biểu:")
                    for idx, c in enumerate(article_contexts[:5], 1):
                        title = c.metadata.get("title") or c.content.split(".")[0]
                        year = c.metadata.get("year")
                        year_str = f" ({year})" if year else ""
                        citations = c.metadata.get("citations")
                        cit_str = f" • *Trích dẫn: {citations}*" if citations is not None else ""
                        raw_content = c.content
                        if raw_content.startswith(title):
                            raw_content = raw_content[len(title):].lstrip(" .:-")
                        desc = raw_content[:280] + "..." if len(raw_content) > 280 else raw_content
                        sections.append(f"{idx}. **{title}**{year_str}{cit_str}\n   - *Tóm tắt trọng tâm*: {desc}")

                sections.append("\n💡 *Ghi chú: Kết quả được tổng hợp trực tiếp từ cơ sở dữ liệu học thuật PostgreSQL và Đồ thị Tri thức Neo4j.*")
                answer = "\n\n".join(sections)

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
        chat_history_service: Any = None,
        context_memory_service: Any = None,
    ) -> RagPipelineResponse:
        """Executes full in-process End-to-End RAG Pipeline (Steps 7 through 12)."""
        start_overall = time.perf_counter()

        user_message_id = None
        asst_message_id = None

        # Context Memory: Resolve coreferences in follow-up queries and fetch history
        effective_query = request.query
        history_context = None
        if context_memory_service and (request.user_id or request.project_id):
            try:
                effective_query = context_memory_service.reformulate_query_with_context(
                    request.query, request.project_id, request.user_id
                )
                history_context = context_memory_service.format_history_for_prompt(
                    request.project_id, request.user_id, max_turns=3
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Context memory resolution error: %s", e)

        # Step 7-9: Retrieval (In-process call without HTTP overhead)
        t0 = time.perf_counter()
        retrieval_res = retrieval_service.retrieve(
            RetrievalRequest(
                query=effective_query,
                top_k=request.top_k or 5,
                project_id=request.project_id,
            )
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
                metadata=c.metadata,
            )
            for c in retrieval_res.results
        ]
        gen_res = self.generate(
            GenerationRequest(
                query=request.query,
                contexts=contexts_for_gen,
                model=request.model,
                temperature=request.temperature,
                history_context=history_context,
            )
        )
        generation_ms = round((time.perf_counter() - t1) * 1000, 2)
        total_ms = round((time.perf_counter() - start_overall) * 1000, 2)

        # Update Context Memory for conversational continuity
        if context_memory_service and (request.user_id or request.project_id):
            try:
                context_memory_service.record_turn(
                    project_id=request.project_id,
                    user_id=request.user_id,
                    user_query=request.query,
                    assistant_answer=gen_res.answer,
                    contexts=retrieval_res.results,
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Failed to record turn in context memory: %s", e)

        # Persist conversation turn in database if requested
        if request.save_history and request.user_id and chat_history_service:
            try:
                user_message_id, asst_message_id = chat_history_service.record_chat_turn(
                    project_id=request.project_id,
                    user_id=request.user_id,
                    user_query=request.query,
                    assistant_answer=gen_res.answer,
                    model=gen_res.model,
                    prompt_tokens=gen_res.usage.prompt_tokens,
                    completion_tokens=gen_res.usage.completion_tokens,
                    total_tokens=gen_res.usage.total_tokens,
                    latency_ms=int(total_ms),
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Failed to auto-record chat history: %s", e)


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
            user_message_id=user_message_id,
            assistant_message_id=asst_message_id,
        )

