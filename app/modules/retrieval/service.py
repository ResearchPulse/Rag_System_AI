import time
import json
import urllib.request
from typing import Any, List

from app.core.config import get_settings
from app.modules.retrieval.schemas import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)
from app.modules.retrieval.query_classifier import (
    QueryClassifier,
    QueryCategory,
    QueryIntent,
    QuerySubCategory,
)
from app.modules.retrieval.graph_retriever import GraphRetriever


class RetrievalService:
    """Service orchestrating Phase 3: Retrieval with Intelligent Semantic Routing.
    
    Routes queries to:
    - VECTOR: PostgreSQL pgvector cosine similarity search
    - GRAPH: Neo4j Knowledge Graph Cypher traversal
    - HYBRID: Combined relational knowledge graph context + vector retrieval
    - METADATA: Exact identifier lookup (DOI, ID) in PostgreSQL
    - SQL AGGREGATION: Direct statistical counts and metrics
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.classifier = QueryClassifier()
        self.graph_retriever = GraphRetriever()

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Executes intelligent query classification and routed retrieval."""
        start_time = time.perf_counter()
        results: List[RetrievedChunk] = []

        # Step 1: Semantic Classification / Intent Routing
        classification = self.classifier.classify(request.query)
        cat = classification.category
        sub_cat = classification.sub_category
        limit = request.top_k or 5

        # Step 2: Route according to classification
        if cat == QueryCategory.CHITCHAT:
            # Conversational greetings - skip database lookup
            pass

        elif cat == QueryCategory.CLARIFICATION_NEEDED:
            results.append(
                RetrievedChunk(
                    chunk_id="clarification_needed",
                    document_id="doc_clarification",
                    content=(
                        "[Yêu cầu làm rõ thông tin] Câu hỏi của bạn quá ngắn hoặc chưa rõ thực thể cần tra cứu. "
                        "Xin vui lòng bổ sung thêm thông tin cụ thể (ví dụ: tên tác giả, tên tạp chí, "
                        "lĩnh vực/chủ đề nghiên cứu, hoặc năm xuất bản) để ResearchPulse có thể hỗ trợ bạn chính xác nhất."
                    ),
                    score=1.0,
                    rerank_score=1.0,
                    metadata={"source": "system_guardrail", "type": "clarification_needed"},
                )
            )

        elif sub_cat == QuerySubCategory.METADATA_LOOKUP and classification.extracted_filters and classification.extracted_filters.doi:
            doi_chunks = self._lookup_by_doi(classification.extracted_filters.doi)
            if doi_chunks:
                results.extend(doi_chunks)
            else:
                results.extend(self._retrieve_vector(request.query, limit=limit))

        elif classification.execution_plan.requires_sql_aggregation:
            # direct_lookup (SQL count / statistical aggregation)
            sql_chunks = self._execute_sql_aggregation(request.query, classification.extracted_filters)
            results.extend(sql_chunks)

        elif classification.requires_graph_traversal and classification.requires_vector_search:
            # hybrid: relational graph traversal + vector similarity search
            graph_chunks = self.graph_retriever.retrieve(
                request.query,
                classification.entities,
                limit=max(2, limit // 2),
            )
            results.extend(graph_chunks)
            vec_chunks = self._retrieve_vector(request.query, limit=limit - len(graph_chunks))
            results.extend(vec_chunks)

        elif classification.requires_graph_traversal:
            # relational_reasoning: Neo4j graph traversal
            graph_chunks = self.graph_retriever.retrieve(
                request.query,
                classification.entities,
                limit=limit,
            )
            results.extend(graph_chunks)

        else:
            # direct_lookup: PostgreSQL pgvector similarity search
            results.extend(self._retrieve_vector(request.query, limit=limit))

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return RetrievalResponse(
            query=request.query,
            total_found=len(results),
            results=results[:limit],
            latency_ms=latency_ms,
        )

    def _retrieve_vector(self, query: str, limit: int = 5) -> List[RetrievedChunk]:
        """Performs pgvector cosine similarity search on PostgreSQL Article table."""
        if not self.settings.GEMINI_API_KEY:
            return []

        results: List[RetrievedChunk] = []
        try:
            import psycopg2

            # 1. Embed query
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={self.settings.GEMINI_API_KEY}"
            payload = {
                "content": {"parts": [{"text": query[:3000]}]},
                "outputDimensionality": self.settings.EMBEDDING_DIMENSION or 768,
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                q_vec = data["embedding"]["values"]

            # 2. Query PostgreSQL "Article" table
            dsn = (
                f"postgresql://{self.settings.POSTGRES_USER}:{self.settings.POSTGRES_PASSWORD}"
                f"@{self.settings.POSTGRES_HOST}:{self.settings.POSTGRES_PORT}/{self.settings.POSTGRES_DB}"
            )
            conn = psycopg2.connect(dsn, connect_timeout=5)
            cur = conn.cursor()
            vector_str = f"[{','.join(map(str, q_vec))}]"
            threshold = float(self.settings.DEFAULT_SCORE_THRESHOLD or 0.55)
            sql = """
                SELECT article_id, title, COALESCE(abstract, ''), publication_year,
                       1 - (embedding <=> %s::vector) AS score
                FROM "Article"
                WHERE embedding IS NOT NULL AND (1 - (embedding <=> %s::vector)) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            cur.execute(sql, (vector_str, vector_str, threshold, vector_str, limit))
            rows = cur.fetchall()
            conn.close()

            for row in rows:
                aid, title, abstract, year, score = row
                results.append(
                    RetrievedChunk(
                        chunk_id=f"art_{aid}",
                        document_id=f"doc_{aid}",
                        content=f"{title}. {abstract}".strip(),
                        score=round(float(score), 4),
                        rerank_score=round(float(score), 4),
                        metadata={"source": "pgvector", "title": title, "year": year},
                    )
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Vector retrieval exception: {e}")
            results = []

        return results

    def _execute_sql_aggregation(
        self,
        query: str,
        filters: Any = None,
    ) -> List[RetrievedChunk]:
        """Executes exact SQL count aggregation on PostgreSQL Article table."""
        try:
            import psycopg2
            import re

            dsn = (
                f"postgresql://{self.settings.POSTGRES_USER}:{self.settings.POSTGRES_PASSWORD}"
                f"@{self.settings.POSTGRES_HOST}:{self.settings.POSTGRES_PORT}/{self.settings.POSTGRES_DB}"
            )
            conn = psycopg2.connect(dsn, connect_timeout=5)
            cur = conn.cursor()

            year = None
            if filters and getattr(filters, "date_range", None):
                m = re.search(r"\b(19\d\d|20\d\d)\b", filters.date_range)
                if m:
                    year = int(m.group(1))

            if year:
                cur.execute('SELECT COUNT(*) FROM "Article" WHERE publication_year = %s;', (year,))
                count = cur.fetchone()[0]
                content = (
                    f"[Thống kê cơ sở dữ liệu ResearchPulse] "
                    f"Có tổng cộng {count:,} bài báo khoa học được xuất bản trong năm {year}."
                )
            else:
                cur.execute('SELECT COUNT(*) FROM "Article";')
                count = cur.fetchone()[0]
                content = (
                    f"[Thống kê cơ sở dữ liệu ResearchPulse] "
                    f"Có tổng cộng {count:,} bài báo khoa học được ghi nhận trong cơ sở dữ liệu hệ thống."
                )

            conn.close()
            return [
                RetrievedChunk(
                    chunk_id="sql_stat_count",
                    document_id="doc_stat_count",
                    content=content,
                    score=1.0,
                    rerank_score=1.0,
                    metadata={"source": "postgresql_sql", "type": "Aggregation", "count": count, "year": year},
                )
            ]
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"SQL aggregation error: {e}")
            return []

    def _lookup_by_doi(self, doi: str) -> List[RetrievedChunk]:
        """Performs exact or prefix metadata lookup on PostgreSQL Article table using DOI."""
        try:
            import psycopg2

            dsn = (
                f"postgresql://{self.settings.POSTGRES_USER}:{self.settings.POSTGRES_PASSWORD}"
                f"@{self.settings.POSTGRES_HOST}:{self.settings.POSTGRES_PORT}/{self.settings.POSTGRES_DB}"
            )
            conn = psycopg2.connect(dsn, connect_timeout=5)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT article_id, title, COALESCE(abstract, ''), publication_year, doi, citation_count
                FROM "Article"
                WHERE doi ILIKE %s
                LIMIT 1;
                """,
                (f"%{doi}%",),
            )
            row = cur.fetchone()
            conn.close()

            if row:
                aid, title, abstract, year, r_doi, c_count = row
                content = (
                    f"Bài báo [DOI: {r_doi}]: \"{title}\" (Năm xuất bản: {year}, Số trích dẫn: {c_count}). "
                    f"Tóm tắt: {abstract}"
                ).strip()
                return [
                    RetrievedChunk(
                        chunk_id=f"doi_{aid}",
                        document_id=f"doc_{aid}",
                        content=content,
                        score=1.0,
                        rerank_score=1.0,
                        metadata={"source": "postgresql_doi_lookup", "doi": r_doi, "title": title, "year": year},
                    )
                ]
            return []
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"DOI metadata lookup error: {e}")
            return []



