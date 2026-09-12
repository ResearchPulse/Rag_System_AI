"""Enterprise Query Classification & Semantic Routing Layer for ResearchPulse.

Standard Hierarchical Query Classification:
1. direct_lookup:
   - sql_aggregation: Exact counts, group-by, min/max metrics on PostgreSQL SQL.
   - semantic_similarity: Dense vector similarity search on pgVector.
   - metadata_lookup: Exact identifier lookup (DOI, ISSN, exact title).
2. relational_reasoning:
   - co_authorship: Collaborative author networks on Neo4j Graph.
   - author_publications: Works published by a specific scholar.
   - citation_network: Cross-paper citation chains, influential research.
   - venue_indexing: Journal indexing, Quartiles (Q1-Q4), SJR rank.
   - topic_clustering: Taxonomy trees, topic trend hierarchies.
3. hybrid:
   - filtered_graph: Combines metadata filters (year, domain, venue) WITH graph traversal.
4. chitchat:
   - Greetings, conversational pleasantries, or questions outside research domain.
5. clarification_needed:
   - Queries too vague or missing key entities to retrieve accurately.
"""

import json
import logging
import re
import time
import urllib.request
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class QueryCategory(str, Enum):
    DIRECT_LOOKUP = "direct_lookup"
    RELATIONAL_REASONING = "relational_reasoning"
    HYBRID = "hybrid"
    CHITCHAT = "chitchat"
    CLARIFICATION_NEEDED = "clarification_needed"


# Backwards compatibility alias
QueryIntent = QueryCategory


class QuerySubCategory(str, Enum):
    SQL_AGGREGATION = "sql_aggregation"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    METADATA_LOOKUP = "metadata_lookup"
    CO_AUTHORSHIP = "co_authorship"
    AUTHOR_PUBLICATIONS = "author_publications"
    CITATION_NETWORK = "citation_network"
    VENUE_INDEXING = "venue_indexing"
    TOPIC_CLUSTERING = "topic_clustering"
    FILTERED_GRAPH = "filtered_graph"
    CHITCHAT = "chitchat"
    AMBIGUOUS = "ambiguous"


class TargetStore(str, Enum):
    POSTGRESQL_SQL = "postgresql_sql"
    POSTGRESQL_PGVECTOR = "postgresql_pgvector"
    NEO4J_GRAPH = "neo4j_graph"
    HYBRID_ALL = "hybrid_all"
    NONE = "none"


class ExtractedFilters(BaseModel):
    date_range: Optional[str] = Field(default=None, description="Date or year range, e.g. '2023-01-01_to_2023-12-31' or '>=2020'")
    year: Optional[int] = Field(default=None, description="Extracted single target publication year")
    subject_category: Optional[str] = Field(default=None, description="Subject category / field of research")
    keyword: Optional[str] = Field(default=None, description="Key research concepts or terms")
    author: Optional[str] = Field(default=None, description="Author name(s)")
    journal: Optional[str] = Field(default=None, description="Journal or publisher name")
    doi: Optional[str] = Field(default=None, description="Extracted Digital Object Identifier (DOI)")
    metric: Optional[str] = Field(default=None, description="Statistical metric requirement (e.g. count, citations, rank)")


class ExecutionPlan(BaseModel):
    requires_sql_aggregation: bool = False
    requires_vector_search: bool = False
    requires_graph_traversal: bool = False
    target_store: TargetStore = TargetStore.NONE
    rewritten_query: Optional[str] = None
    suggested_cypher: Optional[str] = Field(default=None, description="Suggested Cypher query template for Neo4j")
    suggested_sql: Optional[str] = Field(default=None, description="Suggested SQL query template for PostgreSQL")
    recommended_retrievers: List[str] = Field(default_factory=list)
    top_k: int = 5


class ClassificationResult(BaseModel):
    category: QueryCategory
    sub_category: QuerySubCategory = QuerySubCategory.SEMANTIC_SIMILARITY
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Classification confidence (0.0 - 1.0)")
    reasoning: str = ""
    execution_plan: ExecutionPlan = Field(default_factory=ExecutionPlan)
    extracted_filters: Optional[ExtractedFilters] = None
    detected_language: str = Field(default="vi", description="'vi', 'en', or 'other'")
    classification_engine: str = Field(default="llm_gemini", description="Engine used: fastpath_rule, llm_gemini, or heuristic_fallback")
    latency_ms: float = 0.0

    @property
    def intent(self) -> QueryCategory:
        return self.category

    @property
    def requires_graph_traversal(self) -> bool:
        return self.execution_plan.requires_graph_traversal

    @property
    def requires_vector_search(self) -> bool:
        return self.execution_plan.requires_vector_search

    @property
    def entities(self) -> Dict[str, Any]:
        """Convenience property mapping extracted_filters to dictionary for graph/vector retrievers."""
        if not self.extracted_filters:
            return {}
        res: Dict[str, Any] = {}
        if self.extracted_filters.author:
            res["authors"] = [a.strip() for a in self.extracted_filters.author.split(",") if a.strip()]
        if self.extracted_filters.subject_category:
            res["topics"] = [self.extracted_filters.subject_category]
        if self.extracted_filters.keyword:
            res["keywords"] = [self.extracted_filters.keyword]
        if self.extracted_filters.journal:
            res["journals"] = [self.extracted_filters.journal]
        if self.extracted_filters.date_range:
            res["date_range"] = self.extracted_filters.date_range
        if self.extracted_filters.year:
            res["year"] = self.extracted_filters.year
        if self.extracted_filters.doi:
            res["doi"] = self.extracted_filters.doi
        return res


CLASSIFIER_PROMPT = """Bạn là bộ phân loại truy vấn (Query Classifier) cấp doanh nghiệp trong hệ thống RAG cho nền tảng theo dõi xu hướng xuất bản khoa học (ResearchPulse).
Nhiệm vụ DUY NHẤT của bạn là phân tích câu hỏi người dùng, phân loại vào ĐÚNG MỘT trong các nhóm chuẩn mực dưới đây, trích xuất thực thể/bộ lọc và lập kế hoạch định tuyến (Execution Plan).
TUYỆT ĐỐI KHÔNG trả lời trực tiếp nội dung câu hỏi.

## HỆ THỐNG PHÂN LOẠI (TAXONOMY)

1. "direct_lookup":
   Tra cứu dữ liệu đơn giản trên PostgreSQL bằng SQL hoặc pgVector.
   - sub_category:
     * "sql_aggregation": Đếm số lượng bài báo, thống kê theo năm, tính trung bình (VD: "Có bao nhiêu bài báo năm 2023?", "Thống kê số lượng bài báo từ 2020 đến 2024").
     * "semantic_similarity": Tìm kiếm bài báo tương đồng về nội dung, khái niệm kỹ thuật, tóm tắt bài báo (VD: "Tìm bài báo về Transformer trong xử lý ảnh", "RAG architectures in healthcare").
     * "metadata_lookup": Tra cứu chính xác theo mã định danh DOI, ISSN, hoặc tiêu đề chuẩn (VD: "Tìm bài báo có DOI 10.1016/j.procs.2023.01").

2. "relational_reasoning":
   Suy luận qua nhiều bước dựa trên MỐI QUAN HỆ giữa các thực thể trong Đồ thị tri thức (Neo4j Graph).
   - sub_category:
     * "co_authorship": Mạng lưới đồng tác giả, ai từng hợp tác viết bài với ai (VD: "Tác giả X đã hợp tác với ai?", "Mạng lưới đồng tác giả của GS Y").
     * "author_publications": Toàn bộ bài báo hoặc lịch sử xuất bản của tác giả cụ thể (VD: "Tác giả Xue Qin Yu đã công bố những bài báo nào?").
     * "citation_network": Quan hệ trích dẫn, bài báo nào trích dẫn bài báo nào, bài báo ảnh hưởng nhất (VD: "Những bài báo nào trích dẫn công trình của Hinton 2017?").
     * "venue_indexing": Tạp chí, xếp hạng Q1/Q2/Q3/Q4, chỉ số SJR, nhà xuất bản (VD: "Tạp chí IEEE Access thuộc Quartile mấy?").
     * "topic_clustering": Cây chủ đề nghiên cứu, mối quan hệ giữa các lĩnh vực học thuật.

3. "hybrid":
   Kết hợp CẢ HAI: Vừa có bộ lọc siêu dữ liệu (năm, chủ đề, từ khóa, xếp hạng) VÀ vừa cần duyệt quan hệ đồ thị (Graph Traversal).
   - sub_category:
     * "filtered_graph": Duyệt đồ thị trên tập bài báo đã lọc theo năm/chủ đề (VD: "Trong các bài báo xuất bản sau 2023 thuộc chủ đề AI, tác giả nào hợp tác với nhau nhiều nhất?").

4. "chitchat":
   Chào hỏi, giới thiệu bản thân bot, câu hỏi xã giao hoặc hoàn toàn ngoài phạm vi dữ liệu nghiên cứu khoa học (VD: "Chào bạn", "Thời tiết hôm nay thế nào?", "Bạn có thể làm gì?").
   - sub_category: "chitchat".

5. "clarification_needed":
   Câu hỏi quá mơ hồ, quá ngắn hoặc cụt ngủn, thiếu thực thể rõ ràng để hệ thống có thể truy xuất (VD: "bài báo", "tác giả", "2023", "cho tôi xem đi").
   - sub_category: "ambiguous".

## QUY TẮC PHÂN LOẠI
- Nếu có DOI rõ ràng -> "direct_lookup", sub_category: "metadata_lookup".
- Nếu câu hỏi hỏi số lượng ("bao nhiêu", "how many", "thống kê số") -> "direct_lookup", sub_category: "sql_aggregation".
- Nếu câu hỏi có từ khóa quan hệ (hợp tác, đồng tác giả, trích dẫn, ảnh hưởng, mạng lưới) VÀ có điều kiện năm/chủ đề -> "hybrid", sub_category: "filtered_graph".
- Nếu câu hỏi thuần túy về mối quan hệ tác giả/trích dẫn -> "relational_reasoning".
- Nếu câu chào hỏi hoặc câu ngoài chuyên môn -> "chitchat".
- Nếu câu quá ngắn không đủ ý nghĩa -> "clarification_needed".

## ĐỊNH DẠNG ĐẦU RA - CHỈ TRẢ VỀ DUY NHẤT MỘT JSON OBJECT:
{
  "category": "direct_lookup" | "relational_reasoning" | "hybrid" | "chitchat" | "clarification_needed",
  "sub_category": "sql_aggregation" | "semantic_similarity" | "metadata_lookup" | "co_authorship" | "author_publications" | "citation_network" | "venue_indexing" | "topic_clustering" | "filtered_graph" | "chitchat" | "ambiguous",
  "confidence_score": 0.95,
  "detected_language": "vi" | "en",
  "reasoning": "<Giải thích ngắn gọn lý do phân loại>",
  "execution_plan": {
    "requires_sql_aggregation": true | false,
    "requires_vector_search": true | false,
    "requires_graph_traversal": true | false,
    "target_store": "postgresql_sql" | "postgresql_pgvector" | "neo4j_graph" | "hybrid_all" | "none",
    "rewritten_query": "<Câu truy vấn rút gọn, loại bỏ stopwords để tối ưu tìm kiếm>",
    "suggested_cypher": "<Mẫu câu Cypher gợi ý nếu cần graph traversal, hoặc null>",
    "suggested_sql": "<Mẫu câu SQL gợi ý nếu cần SQL aggregation, hoặc null>",
    "recommended_retrievers": ["sql_count_retriever"] | ["graph_retriever"] | ["vector_retriever"] | ["hybrid_retriever"],
    "top_k": 5
  },
  "extracted_filters": {
    "date_range": "<ví dụ '2023-01-01_to_2023-12-31' hoặc null>",
    "year": 2023 | null,
    "subject_category": "<chuyên ngành nếu có, hoặc null>",
    "keyword": "<từ khóa nghiên cứu, hoặc null>",
    "author": "<tên tác giả nếu có, hoặc null>",
    "journal": "<tên tạp chí nếu có, hoặc null>",
    "doi": "<mã DOI nếu có, hoặc null>",
    "metric": "<chỉ số thống kê nếu có, hoặc null>"
  } | null
}
"""


class QueryClassifier:
    """Enterprise 3-Layer Query Classifier for ResearchPulse."""

    CHITCHAT_PATTERNS = [
        r"^(chào|hello|hi|hey|alo|xin chào|good morning|good afternoon)\b",
        r"\b(bạn là ai|who are you|bạn tên gì|bạn làm được gì|giới thiệu|hướng dẫn)\b",
        r"\b(cảm ơn|thank you|thanks|tạm biệt|bye|goodbye)\b",
        r"\b(thời tiết|hôm nay thế nào|weather today|kể chuyện|hát đi)\b",
    ]

    DOI_PATTERN = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")

    COUNT_PATTERNS = [
        r"\b(bao nhiêu bài|how many articles|how many papers|thống kê số lượng bài|đếm số bài)\b",
    ]

    RELATIONAL_KEYWORDS = [
        "hợp tác", "đồng tác giả", "co-author", "collaborat", "ai viết",
        "who wrote", "trích dẫn", "citation", "cites", "mạng lưới trích dẫn",
        "mạng lưới tác giả", "mạng lưới hợp tác", "citation network",
        "quan hệ", "relationship", "liên kết tác giả", "kết nối tác giả", "ảnh hưởng lớn nhất",
    ]

    def __init__(self) -> None:
        self.settings = get_settings()

    def classify(self, query: str) -> ClassificationResult:
        """Executes 3-layer classification pipeline:
        1. Fast-path rule engine (0ms latency, zero token cost)
        2. In-context LLM router (Gemini 2.5 Flash, structured JSON)
        3. Production heuristic fallback (resilient against quota limits)
        """
        t0 = time.perf_counter()
        cleaned_query = query.strip()

        # --- LAYER 1: FAST-PATH RULE ENGINE ---
        fast_result = self._classify_fastpath(cleaned_query)
        if fast_result is not None:
            fast_result.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return fast_result

        # --- LAYER 2: LLM IN-CONTEXT SEMANTIC ROUTER ---
        if self.settings.GEMINI_API_KEY:
            try:
                res = self._classify_with_llm(cleaned_query)
                res.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                return res
            except Exception as e:
                logger.warning(f"LLM classifier failed or rate-limited, switching to Layer 3 Heuristic: {e}")

        # --- LAYER 3: ENTERPRISE HEURISTIC FALLBACK ---
        res = self._classify_heuristic(cleaned_query)
        res.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return res

    def _classify_fastpath(self, query: str) -> Optional[ClassificationResult]:
        """Sub-millisecond rule check for obvious intents to save latency and token quota."""
        lower = query.lower()

        # Check 1: Vague or empty / clarification needed
        words = query.split()
        if len(words) <= 2 and lower not in ["rag", "llm", "ai", "transformer", "bert", "gnn"]:
            if lower in ["bài báo", "tác giả", "paper", "author", "2023", "2024", "cho tôi xem", "nghiên cứu", "xuất bản"]:
                return ClassificationResult(
                    category=QueryCategory.CLARIFICATION_NEEDED,
                    sub_category=QuerySubCategory.AMBIGUOUS,
                    confidence_score=0.99,
                    reasoning="Câu truy vấn quá ngắn và thiếu thực thể, cần làm rõ thông tin chi tiết.",
                    execution_plan=ExecutionPlan(
                        requires_sql_aggregation=False,
                        requires_vector_search=False,
                        requires_graph_traversal=False,
                        target_store=TargetStore.NONE,
                        recommended_retrievers=[],
                    ),
                    extracted_filters=None,
                    detected_language="vi" if re.search(r"[à-ỹ]", lower) else "en",
                    classification_engine="fastpath_rule",
                )

        # Check 2: Chitchat greeting
        for pat in self.CHITCHAT_PATTERNS:
            if re.search(pat, lower):
                # Ensure no academic term is in the query
                if not any(k in lower for k in ["bài báo", "tác giả", "xuất bản", "paper", "author", "doi", "journal"]):
                    return ClassificationResult(
                        category=QueryCategory.CHITCHAT,
                        sub_category=QuerySubCategory.CHITCHAT,
                        confidence_score=0.99,
                        reasoning="Nhận diện câu chào hỏi hoặc giao tiếp xã giao thông qua Fast-path Rule Engine.",
                        execution_plan=ExecutionPlan(
                            requires_sql_aggregation=False,
                            requires_vector_search=False,
                            requires_graph_traversal=False,
                            target_store=TargetStore.NONE,
                            recommended_retrievers=[],
                        ),
                        extracted_filters=None,
                        detected_language="vi" if re.search(r"[à-ỹ]", lower) else "en",
                        classification_engine="fastpath_rule",
                    )

        # Check 3: DOI Lookup pattern
        doi_match = self.DOI_PATTERN.search(query)
        if doi_match and not any(k in lower for k in self.RELATIONAL_KEYWORDS):
            doi = doi_match.group(1)
            return ClassificationResult(
                category=QueryCategory.DIRECT_LOOKUP,
                sub_category=QuerySubCategory.METADATA_LOOKUP,
                confidence_score=0.99,
                reasoning="Nhận diện mã định danh số DOI hợp lệ cho tra cứu tài liệu trực tiếp.",
                execution_plan=ExecutionPlan(
                    requires_sql_aggregation=False,
                    requires_vector_search=False,
                    requires_graph_traversal=False,
                    target_store=TargetStore.POSTGRESQL_SQL,
                    rewritten_query=doi,
                    suggested_sql=f'SELECT article_id, title, abstract FROM "Article" WHERE doi = \'{doi}\' LIMIT 1;',
                    recommended_retrievers=["metadata_lookup_retriever"],
                    top_k=1,
                ),
                extracted_filters=ExtractedFilters(doi=doi),
                detected_language="en",
                classification_engine="fastpath_rule",
            )

        return None

    def _classify_with_llm(self, query: str) -> ClassificationResult:
        """Executes LLM In-Context classification using Gemini 2.5 Flash."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.settings.GEMINI_API_KEY}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"{CLASSIFIER_PROMPT}\n\nInput Query: \"{query}\"\nOutput JSON:"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)

            cat_str = str(parsed.get("category", "direct_lookup")).lower()
            try:
                category = QueryCategory(cat_str)
            except ValueError:
                category = QueryCategory.DIRECT_LOOKUP

            sub_str = str(parsed.get("sub_category", "semantic_similarity")).lower()
            try:
                sub_category = QuerySubCategory(sub_str)
            except ValueError:
                sub_category = QuerySubCategory.SEMANTIC_SIMILARITY

            plan_dict = parsed.get("execution_plan", {})
            target_str = str(plan_dict.get("target_store", "postgresql_pgvector")).lower()
            try:
                target_store = TargetStore(target_str)
            except ValueError:
                target_store = TargetStore.POSTGRESQL_PGVECTOR

            execution_plan = ExecutionPlan(
                requires_sql_aggregation=bool(plan_dict.get("requires_sql_aggregation", False)),
                requires_vector_search=bool(plan_dict.get("requires_vector_search", True)),
                requires_graph_traversal=bool(plan_dict.get("requires_graph_traversal", False)),
                target_store=target_store,
                rewritten_query=plan_dict.get("rewritten_query"),
                suggested_cypher=plan_dict.get("suggested_cypher"),
                suggested_sql=plan_dict.get("suggested_sql"),
                recommended_retrievers=plan_dict.get("recommended_retrievers", []),
                top_k=int(plan_dict.get("top_k", 5)),
            )

            raw_filters = parsed.get("extracted_filters")
            extracted_filters = ExtractedFilters(**raw_filters) if raw_filters else None

            confidence = float(parsed.get("confidence_score", 0.95))
            detected_lang = str(parsed.get("detected_language", "vi"))

            return ClassificationResult(
                category=category,
                sub_category=sub_category,
                confidence_score=confidence,
                reasoning=str(parsed.get("reasoning", "Phân loại thành công qua Gemini LLM Router.")),
                execution_plan=execution_plan,
                extracted_filters=extracted_filters,
                detected_language=detected_lang,
                classification_engine="llm_gemini",
            )

    def _classify_heuristic(self, query: str) -> ClassificationResult:
        """Robust, comprehensive heuristic engine ensuring 100% uptime when LLM is unavailable."""
        lower_q = query.lower()
        detected_lang = "vi" if re.search(r"[à-ỹ]", lower_q) else "en"

        # 1. Year Extraction
        year = None
        date_range = None
        year_match = re.search(r"\b(19\d\d|20\d\d)\b", query)
        if year_match:
            year = int(year_match.group(1))
            if any(k in lower_q for k in ["sau", "sau năm", "after", "since", ">="]):
                date_range = f">={year}"
            elif any(k in lower_q for k in ["trước", "before", "<="]):
                date_range = f"<={year}"
            else:
                date_range = f"{year}-01-01_to_{year}-12-31"

        # 2. DOI Extraction
        doi_match = self.DOI_PATTERN.search(query)
        doi = doi_match.group(1) if doi_match else None

        # 3. Author Extraction
        author = None
        author_match = re.search(
            r"(?:tác giả|author|by)\s+([^\d\?,\.;\n]+?)(?:\s+(?:đã|có|là|nào|trong|vào|và|hợp tác|thường|hay|cùng|who|has|published)|[\?,\.;]|$)",
            query,
            re.IGNORECASE,
        )
        if author_match:
            raw = author_match.group(1).strip()
            if raw.lower() not in ["nào", "ai", "mấy", "gì", "những ai", "bao nhiêu"]:
                caps = re.findall(r"\b[A-Z][a-zA-Z]*\b", raw)
                author = " ".join(caps) if caps else raw

        # 4. Journal / Venue Extraction
        journal = None
        journal_match = re.search(r"(?:tạp chí|journal|venue)\s+([^\d\?,\.;\n]+?)(?:\s+(?:đã|có|xuất bản|published)|[\?,\.;]|$)", query, re.IGNORECASE)
        if journal_match:
            journal = journal_match.group(1).strip()

        # 5. Subject / Topic Extraction
        subject = None
        topic_match = re.search(r"(?:lĩnh vực|chủ đề|topic|chuyên ngành)\s+([^\d\?,\.;\n]+?)(?:\s+(?:có|trong|sau|before|after)|[\?,\.;]|$)", query, re.IGNORECASE)
        if topic_match:
            subject = topic_match.group(1).strip()

        # 6. Keyword Extraction
        keyword = None
        kw_match = re.search(r"(?:về|about|concept)\s+([^\d\?,\.;\n]+?)(?:\s+(?:trong|in|tại)|[\?,\.;]|$)", query, re.IGNORECASE)
        if kw_match:
            raw_kw = kw_match.group(1).strip()
            if raw_kw.lower() not in ["bài báo", "nghiên cứu", "khoa học"]:
                keyword = raw_kw

        extracted_filters = ExtractedFilters(
            date_range=date_range,
            year=year,
            subject_category=subject,
            keyword=keyword,
            author=author,
            journal=journal,
            doi=doi,
        )

        has_relational = any(kw in lower_q for kw in self.RELATIONAL_KEYWORDS)
        is_count_query = any(re.search(pat, lower_q) for pat in self.COUNT_PATTERNS)

        # CASE 1: SQL Aggregation (direct_lookup)
        if is_count_query and not has_relational:
            sql_hint = f'SELECT COUNT(*) FROM "Article" WHERE publication_year = {year};' if year else 'SELECT COUNT(*) FROM "Article";'
            return ClassificationResult(
                category=QueryCategory.DIRECT_LOOKUP,
                sub_category=QuerySubCategory.SQL_AGGREGATION,
                confidence_score=0.95,
                reasoning="Câu hỏi thống kê số lượng bài báo đơn giản, tối ưu định tuyến trực tiếp vào PostgreSQL SQL Aggregator.",
                execution_plan=ExecutionPlan(
                    requires_sql_aggregation=True,
                    requires_vector_search=False,
                    requires_graph_traversal=False,
                    target_store=TargetStore.POSTGRESQL_SQL,
                    rewritten_query=query,
                    suggested_sql=sql_hint,
                    recommended_retrievers=["sql_count_retriever"],
                    top_k=1,
                ),
                extracted_filters=extracted_filters,
                detected_language=detected_lang,
                classification_engine="heuristic_fallback",
            )

        # CASE 2: Hybrid (Filtered conditions + Graph Traversal)
        if has_relational and (year or subject or keyword):
            cypher_hint = (
                "MATCH (a1:Author)-[:AUTHORED]->(art:Article)<-[:AUTHORED]-(a2:Author) "
                f"WHERE art.publication_year >= {year or 2020} "
                "RETURN a1.name, a2.name, count(art) AS collaborations ORDER BY collaborations DESC LIMIT 5"
            )
            return ClassificationResult(
                category=QueryCategory.HYBRID,
                sub_category=QuerySubCategory.FILTERED_GRAPH,
                confidence_score=0.92,
                reasoning="Truy vấn kết hợp điều kiện lọc (năm/chủ đề) VÀ suy luận quan hệ hợp tác đồ thị tri thức.",
                execution_plan=ExecutionPlan(
                    requires_sql_aggregation=False,
                    requires_vector_search=True,
                    requires_graph_traversal=True,
                    target_store=TargetStore.HYBRID_ALL,
                    rewritten_query=query,
                    suggested_cypher=cypher_hint,
                    recommended_retrievers=["graph_retriever", "vector_retriever"],
                    top_k=5,
                ),
                extracted_filters=extracted_filters,
                detected_language=detected_lang,
                classification_engine="heuristic_fallback",
            )

        # CASE 3: Relational Reasoning (Pure Graph Traversal)
        if has_relational or author:
            is_coauthor = any(k in lower_q for k in ["hợp tác", "đồng tác giả", "co-author", "collaborat"])
            sub = QuerySubCategory.CO_AUTHORSHIP if is_coauthor else QuerySubCategory.AUTHOR_PUBLICATIONS
            cypher_hint = (
                f"MATCH (a:Author {{name: '{author or 'Target'}'}})-[:AUTHORED]->(art:Article) RETURN art.title, art.publication_year LIMIT 10"
            )
            return ClassificationResult(
                category=QueryCategory.RELATIONAL_REASONING,
                sub_category=sub,
                confidence_score=0.93,
                reasoning="Truy vấn cần phân tích các mối quan hệ đa bước trên Đồ thị tri thức Neo4j.",
                execution_plan=ExecutionPlan(
                    requires_sql_aggregation=False,
                    requires_vector_search=False,
                    requires_graph_traversal=True,
                    target_store=TargetStore.NEO4J_GRAPH,
                    rewritten_query=query,
                    suggested_cypher=cypher_hint,
                    recommended_retrievers=["graph_retriever"],
                    top_k=5,
                ),
                extracted_filters=extracted_filters,
                detected_language=detected_lang,
                classification_engine="heuristic_fallback",
            )

        # CASE 4: Direct Lookup (Semantic Similarity on pgVector)
        return ClassificationResult(
            category=QueryCategory.DIRECT_LOOKUP,
            sub_category=QuerySubCategory.SEMANTIC_SIMILARITY,
            confidence_score=0.90,
            reasoning="Truy vấn tìm kiếm nội dung khoa học ngữ nghĩa trên cơ sở dữ liệu vector pgVector.",
            execution_plan=ExecutionPlan(
                requires_sql_aggregation=False,
                requires_vector_search=True,
                requires_graph_traversal=False,
                target_store=TargetStore.POSTGRESQL_PGVECTOR,
                rewritten_query=query,
                recommended_retrievers=["vector_retriever"],
                top_k=5,
            ),
            extracted_filters=extracted_filters,
            detected_language=detected_lang,
            classification_engine="heuristic_fallback",
        )

