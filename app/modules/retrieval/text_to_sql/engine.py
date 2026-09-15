"""Enterprise Text-to-SQL (NL2SQL) Engine for ResearchPulse.

Converts natural language queries directly into PostgreSQL SQL queries with:
- Full schema awareness & case-sensitive table quoting
- Strict project scoping enforcement (Project_Article_Scope)
- Security guardrails (Read-only SELECT, anti-injection, forced LIMIT)
- 1-shot self-correction loop on SQL execution errors
- Markdown table/summary result formatting
"""
import json
import logging
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from app.core.config import get_settings
from app.modules.retrieval.schemas import RetrievedChunk
from app.modules.retrieval.text_to_sql.schemas import (
    SQLExecutionResult,
    SQLValidationResult,
)

logger = logging.getLogger(__name__)

POSTGRESQL_SCHEMA_PROMPT = """Bạn là chuyên gia cơ sở dữ liệu PostgreSQL (PostgreSQL Text-to-SQL Engine) cho hệ thống ResearchPulse.
Nhiệm vụ của bạn là dịch câu hỏi tiếng Việt/tiếng Anh của người dùng thành DUY NHẤT một câu lệnh SQL (SELECT) hợp lệ, chuẩn xác và an toàn.

### LƯỢC ĐỒ CƠ SỞ DỮ LIỆU POSTGRESQL:
Tất cả bảng đều dùng chữ hoa chữ thường có dấu nháy kép:
1. "Article": article_id (bigint, PK), title (text), abstract (text), citation_count (int), publication_year (int), doi (text), primary_topic (bigint FK), issue_id (bigint FK)
2. "Author": author_id (bigint, PK), display_name (text), h_index (int), cited_by_count (int), works_count (int), last_known_institution (text)
3. "Author_Article": author_id (bigint, FK), article_id (bigint, FK), author_position (int)
4. "Journal": journal_id (bigint, PK), display_name (text), country (bigint, FK to Zone.zone_id)
5. "Volume": volume_id (bigint, PK), journal_id (bigint, FK), volume_number (text)
6. "Issue": issue_id (bigint, PK), volume_id (bigint, FK), issue_number (text)
7. "Topic": topic_id (bigint, PK), display_name (text)
8. "Keyword": keyword_id (bigint, PK), display_name (text)
9. "Keyword_Article": keyword_id (bigint, FK), article_id (bigint, FK)
10. "Zone": zone_id (bigint, PK), name (text), code (text), type (text, ví dụ 'COUNTRY')
11. "Project_Article_Scope": project_id (bigint), article_id (bigint) (Lưu danh sách bài báo thuộc phạm vi đề tài)

### QUY TẮC BẮT BUỘC:
1. CHỈ tạo câu lệnh SELECT. Tuyệt đối không dùng INSERT, UPDATE, DELETE, DROP, ALTER.
2. Tên bảng PHẢI có dấu nháy kép: "Article", "Author", "Author_Article", "Journal", "Volume", "Issue", "Topic", "Keyword", "Keyword_Article", "Zone", "Project_Article_Scope".
3. Tên tác giả, tạp chí, chủ đề, từ khóa dùng cột `display_name` (KHÔNG dùng cột `name`, chỉ bảng "Zone" mới có cột `name`).
4. Nếu có PROJECT_ID ({project_id}), BẮT BUỘC phải lọc theo phạm vi đề tài qua bảng "Project_Article_Scope":
   JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id WHERE pas.project_id = {project_id}
5. Với các truy vấn danh sách hoặc xếp hạng, LUÔN thêm `LIMIT 5` hoặc `LIMIT 10`.
6. Dùng `COALESCE(SUM(a.citation_count), 0)` khi tính tổng trích dẫn của tác giả.
7. PHÂN BIỆT TUYỆT ĐỐI GIỮA ĐẾM SỐ LƯỢNG TÁC GIẢ VÀ XẾP HẠNG TÁC GIẢ:
   - Nếu câu hỏi hỏi SỐ LƯỢNG TÁC GIẢ ("số lượng tác giả", "tổng số tác giả", "bao nhiêu tác giả", "count authors"):
     BẮT BUỘC dùng `COUNT(DISTINCT aa.author_id) AS total_authors` (KHÔNG ĐƯỢC GROUP BY tên tác giả, KHÔNG ĐƯỢC đếm số bài báo của từng người).
   - Nếu câu hỏi hỏi ĐỒNG THỜI CẢ TÁC GIẢ VÀ BÀI BÁO ("tổng số lượng tác giả và bài báo", "bao nhiêu tác giả và bao nhiêu bài báo"):
     BẮT BUỘC dùng:
     `SELECT COUNT(DISTINCT aa.author_id) AS total_authors, COUNT(DISTINCT pas.article_id) AS total_articles FROM "Project_Article_Scope" pas LEFT JOIN "Author_Article" aa ON pas.article_id = aa.article_id WHERE pas.project_id = {project_id};`
   - Chỉ khi câu hỏi hỏi "TÁC GIẢ NÀO", "AI", "DẪN ĐẦU", "TOP" có nhiều bài báo/trích dẫn nhất mới GROUP BY au.author_id, au.display_name.
8. QUY TẮC TÊN QUỐC GIA (BẢNG Zone): Cột `name` trong bảng "Zone" LƯU BẰNG TIẾNG ANH VÀ CÓ CỘT `code` (ISO-2)!
   - Khi câu hỏi bằng tiếng Việt nhắc đến quốc gia, BẮT BUỘC map sang tên tiếng Anh hoặc mã code:
     * "Mỹ" / "Hoa Kỳ" / "USA" -> (z.name = 'United States' OR z.code = 'US')
     * "Anh" / "Vương quốc Anh" / "UK" -> (z.name = 'United Kingdom' OR z.code = 'GB')
     * "Việt Nam" / "VN" -> (z.name ILIKE '%Viet%Nam%' OR z.code = 'VN')
     * "Trung Quốc" -> (z.name = 'China' OR z.code = 'CN')
     * "Nhật Bản" -> (z.name = 'Japan' OR z.code = 'JP')
     * "Hàn Quốc" -> (z.name = 'South Korea' OR z.code = 'KR')
     * "Đức" -> (z.name = 'Germany' OR z.code = 'DE')
     * "Pháp" -> (z.name = 'France' OR z.code = 'FR')
     * "Hà Lan" -> (z.name = 'Netherlands' OR z.code = 'NL')
     * "Thụy Sĩ" -> (z.name = 'Switzerland' OR z.code = 'CH')
     * "Tây Ban Nha" -> (z.name = 'Spain' OR z.code = 'ES')
   - TUYỆT ĐỐI KHÔNG để `z.name = 'Mỹ'` hoặc `z.name = 'Hoa Kỳ'` vì trong DB lưu tên là 'United States'!
9. TRẢ VỀ DUY NHẤT CÂU LỆNH SQL, KHÔNG GIẢI THÍCH, KHÔNG CHÈM VĂN BẢN KHÁC.

### VÍ DỤ MẪU:
- Câu hỏi: "Mỹ có bao nhiêu bài báo?" (project_id = {pid})
SQL:
SELECT COUNT(DISTINCT a.article_id) AS total_articles
FROM "Article" a
JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id
JOIN "Issue" i ON a.issue_id = i.issue_id
JOIN "Volume" v ON i.volume_id = v.volume_id
JOIN "Journal" j ON v.journal_id = j.journal_id
JOIN "Zone" z ON j.country = z.zone_id AND z.type = 'COUNTRY'
WHERE (z.name = 'United States' OR z.code = 'US') AND pas.project_id = {pid};

- Câu hỏi: "Số lượng tác giả trong project này là bao nhiêu?" (project_id = {pid})
SQL:
SELECT COUNT(DISTINCT aa.author_id) AS total_authors
FROM "Author_Article" aa
JOIN "Project_Article_Scope" pas ON aa.article_id = pas.article_id
WHERE pas.project_id = {pid};

- Câu hỏi: "Tổng số lượng tác giả và tổng số lượng bài báo trong project này là bao nhiêu ?" (project_id = {pid})
SQL:
SELECT COUNT(DISTINCT aa.author_id) AS total_authors, COUNT(DISTINCT pas.article_id) AS total_articles
FROM "Project_Article_Scope" pas
LEFT JOIN "Author_Article" aa ON pas.article_id = aa.article_id
WHERE pas.project_id = {pid};

- Câu hỏi: "Tổng số tác giả trong dự án/đề tài là bao nhiêu?" (project_id = {pid})
SQL:
SELECT COUNT(DISTINCT aa.author_id) AS total_authors
FROM "Author_Article" aa
JOIN "Project_Article_Scope" pas ON aa.article_id = pas.article_id
WHERE pas.project_id = {pid};

- Câu hỏi: "Có bao nhiêu tác giả trong hệ thống?" (project_id = None)
SQL:
SELECT COUNT(*) AS total_authors FROM "Author";

- Câu hỏi: "Tác giả nào có số lượng bài báo cao nhất trong project này?" (project_id = {pid})
SQL:
SELECT au.display_name, COUNT(DISTINCT aa.article_id) AS paper_count
FROM "Author" au
JOIN "Author_Article" aa ON au.author_id = aa.author_id
JOIN "Project_Article_Scope" pas ON aa.article_id = pas.article_id
WHERE pas.project_id = {pid}
GROUP BY au.author_id, au.display_name
ORDER BY paper_count DESC LIMIT 5;

- Câu hỏi: "Tác giả nào có lượng trích dẫn cao nhất?" (project_id = {pid})
SQL:
SELECT au.display_name, COALESCE(SUM(a.citation_count), 0) AS total_citations, COUNT(DISTINCT a.article_id) AS paper_count, au.h_index, au.last_known_institution
FROM "Author" au
JOIN "Author_Article" aa ON au.author_id = aa.author_id
JOIN "Article" a ON aa.article_id = a.article_id
JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id
WHERE pas.project_id = {pid}
GROUP BY au.author_id, au.display_name, au.h_index, au.last_known_institution
ORDER BY total_citations DESC LIMIT 5;

- Câu hỏi: "Bài báo nào được trích dẫn nhiều nhất?" (project_id = {pid})
SQL:
SELECT a.title, a.publication_year, COALESCE(a.citation_count, 0) AS citations, a.doi
FROM "Article" a
JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id
WHERE pas.project_id = {pid}
ORDER BY a.citation_count DESC NULLS LAST LIMIT 5;

- Câu hỏi: "Tạp chí nào có nhiều bài báo nhất?" (project_id = {pid})
SQL:
SELECT j.display_name, COUNT(DISTINCT a.article_id) AS paper_count
FROM "Journal" j
JOIN "Volume" v ON j.journal_id = v.journal_id
JOIN "Issue" i ON v.volume_id = i.volume_id
JOIN "Article" a ON i.issue_id = a.issue_id
JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id
WHERE pas.project_id = {pid}
GROUP BY j.journal_id, j.display_name
ORDER BY paper_count DESC LIMIT 5;

- Câu hỏi: "Chủ đề nào phổ biến nhất?" (project_id = {pid})
SQL:
SELECT t.display_name, COUNT(DISTINCT a.article_id) AS paper_count
FROM "Topic" t
JOIN "Article" a ON t.topic_id = a.primary_topic
JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id
WHERE pas.project_id = {pid}
GROUP BY t.topic_id, t.display_name
ORDER BY paper_count DESC LIMIT 5;

- Câu hỏi: "Từ khóa nào xuất hiện nhiều nhất?" (project_id = {pid})
SQL:
SELECT k.display_name, COUNT(DISTINCT ka.article_id) AS paper_count
FROM "Keyword" k
JOIN "Keyword_Article" ka ON k.keyword_id = ka.keyword_id
JOIN "Project_Article_Scope" pas ON ka.article_id = pas.article_id
WHERE pas.project_id = {pid}
GROUP BY k.keyword_id, k.display_name
ORDER BY paper_count DESC LIMIT 5;

- Câu hỏi: "Quốc gia nào có nhiều bài báo nhất?" (project_id = {pid})
SQL:
SELECT z.name AS country_name, z.code AS country_code, COUNT(DISTINCT a.article_id) AS paper_count
FROM "Article" a
JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id
JOIN "Issue" i ON a.issue_id = i.issue_id
JOIN "Volume" v ON i.volume_id = v.volume_id
JOIN "Journal" j ON v.journal_id = j.journal_id
JOIN "Zone" z ON j.country = z.zone_id AND z.type = 'COUNTRY'
WHERE pas.project_id = {pid}
GROUP BY z.name, z.code
ORDER BY paper_count DESC LIMIT 5;

- Câu hỏi: "Có bao nhiêu bài báo?" (project_id = {pid})
SQL:
SELECT COUNT(*) AS total_articles FROM "Project_Article_Scope" WHERE project_id = {pid};

- Câu hỏi: "Có bao nhiêu tác giả?" (project_id = {pid})
SQL:
SELECT COUNT(DISTINCT aa.author_id) AS total_authors
FROM "Author_Article" aa
JOIN "Project_Article_Scope" pas ON aa.article_id = pas.article_id
WHERE pas.project_id = {pid};
"""


class TextToSQLEngine:
    """Enterprise dynamic Text-to-SQL execution engine."""

    DISALLOWED_KEYWORDS = [
        r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bDROP\b",
        r"\bALTER\b", r"\bTRUNCATE\b", r"\bCREATE\b", r"\bGRANT\b",
        r"\bREVOKE\b", r"\bEXEC\b", r"\bEXECUTE\b", r"\bCOPY\b",
    ]

    def __init__(self) -> None:
        self.settings = get_settings()

    def _get_postgres_connection(self):
        """Creates connection to PostgreSQL using fallback hosts."""
        hosts_to_try = []
        if self.settings.POSTGRES_HOST:
            hosts_to_try.append(self.settings.POSTGRES_HOST)
        for h in ["100.121.61.95", "127.0.0.1", "localhost"]:
            if h not in hosts_to_try:
                hosts_to_try.append(h)

        for host in hosts_to_try:
            try:
                return psycopg2.connect(
                    host=host,
                    port=self.settings.POSTGRES_PORT,
                    user=self.settings.POSTGRES_USER,
                    password=self.settings.POSTGRES_PASSWORD,
                    dbname=self.settings.POSTGRES_DB,
                    connect_timeout=3,
                )
            except Exception:
                continue
        return None

    def generate_sql(self, query: str, project_id: Optional[int] = None) -> str:
        """Calls LLM (Ollama / Gemini) to dynamically generate SQL from natural language."""
        pid_desc = str(project_id) if project_id else "null (toàn hệ thống)"
        pid_val = str(project_id) if project_id else "1"
        sys_prompt = POSTGRESQL_SCHEMA_PROMPT.replace("{project_id}", pid_desc).replace("{pid}", pid_val)

        user_content = f"PROJECT_ID: {pid_desc}\nCÂU HỎI: {query}\nSQL:"

        # 1. Ollama LLM
        if self.settings.LLM_PROVIDER == "ollama" or getattr(self.settings, "OLLAMA_BASE_URL", None):
            try:
                ollama_url = f"{self.settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
                payload = {
                    "model": self.settings.OLLAMA_MODEL or "llama3.2:3b",
                    "prompt": f"{sys_prompt}\n\n{user_content}",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for precise code generation
                        "num_predict": 256,
                    },
                }
                req = urllib.request.Request(
                    ollama_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    raw_sql = data.get("response", "").strip()
                    clean_sql = self._clean_sql(raw_sql)
                    if clean_sql:
                        return clean_sql
            except Exception as e:
                logger.warning(f"[TextToSQL] Ollama generation failed: {e}")

        # 2. Gemini Fallback
        if self.settings.GEMINI_API_KEY:
            try:
                gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.settings.GEMINI_API_KEY}"
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": f"{sys_prompt}\n\n{user_content}"}],
                        }
                    ],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": 256},
                }
                req = urllib.request.Request(
                    gemini_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates:
                        raw_sql = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        clean_sql = self._clean_sql(raw_sql)
                        if clean_sql:
                            return clean_sql
            except Exception as e:
                logger.warning(f"[TextToSQL] Gemini fallback failed: {e}")

        # Safe fallback template if LLM is unavailable
        return self._generate_fallback_sql(query, project_id)

    def _clean_sql(self, text: str) -> str:
        """Strips markdown code blocks, conversational filler, and trailing semicolons."""
        if not text:
            return ""
        # Match ```sql ... ``` or ``` ... ```
        m = re.search(r"```(?:sql)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
        if m:
            text = m.group(1)
        # Find start of SELECT
        sel_idx = text.upper().find("SELECT")
        if sel_idx >= 0:
            text = text[sel_idx:]
        # Remove trailing comments or semicolons
        text = re.sub(r";+\s*$", "", text.strip())
        return text.strip()

    def _generate_fallback_sql(self, query: str, project_id: Optional[int]) -> str:
        """Deterministic fallback SQL when LLM connection is unavailable."""
        lower = query.lower()
        has_author = "tác giả" in lower or "author" in lower
        has_article = any(k in lower for k in ["bài báo", "bai bao", "paper", "công trình", "article"])
        is_count = any(k in lower for k in ["bao nhiêu", "bao nhieu", "số lượng", "so luong", "tổng số", "tong so", "mấy", "how many", "count", "total"])

        # Multi-entity count: both author and article counts requested
        if has_author and has_article and is_count and not any(k in lower for k in ["nhiều nhất", "nhieu nhat", "cao nhất", "cao nhat", "dẫn đầu", "top", "trích dẫn"]):
            if project_id:
                return (
                    f'SELECT COUNT(DISTINCT aa.author_id) AS total_authors, COUNT(DISTINCT pas.article_id) AS total_articles '
                    f'FROM "Project_Article_Scope" pas '
                    f'LEFT JOIN "Author_Article" aa ON pas.article_id = aa.article_id '
                    f'WHERE pas.project_id = {project_id};'
                )
            return 'SELECT (SELECT COUNT(*) FROM "Author") AS total_authors, (SELECT COUNT(*) FROM "Article") AS total_articles;'

        if has_author:
            is_paper_ranking = any(k in lower for k in ["bài báo", "bai bao", "paper", "công trình"]) and any(k in lower for k in ["nhiều nhất", "nhieu nhat", "cao nhất", "cao nhat", "dẫn đầu", "top"])
            is_citation_ranking = any(k in lower for k in ["trích dẫn", "trich dan", "citation", "cited", "h-index", "h index"])

            if is_count and not (is_paper_ranking or is_citation_ranking):
                if project_id:
                    return f'SELECT COUNT(DISTINCT aa.author_id) AS total_authors FROM "Author_Article" aa JOIN "Project_Article_Scope" pas ON aa.article_id = pas.article_id WHERE pas.project_id = {project_id};'
                return 'SELECT COUNT(*) AS total_authors FROM "Author";'

            if is_paper_ranking:
                if project_id:
                    return f'SELECT au.display_name, COUNT(DISTINCT aa.article_id) AS paper_count FROM "Author" au JOIN "Author_Article" aa ON au.author_id = aa.author_id JOIN "Project_Article_Scope" pas ON aa.article_id = pas.article_id WHERE pas.project_id = {project_id} GROUP BY au.author_id, au.display_name ORDER BY paper_count DESC LIMIT 5;'
                return 'SELECT au.display_name, COUNT(DISTINCT aa.article_id) AS paper_count FROM "Author" au JOIN "Author_Article" aa ON au.author_id = aa.author_id GROUP BY au.author_id, au.display_name ORDER BY paper_count DESC LIMIT 5;'

            if project_id:
                return (
                    'SELECT au.display_name, COALESCE(SUM(a.citation_count), 0) AS total_citations, COUNT(DISTINCT a.article_id) AS paper_count, au.h_index, au.last_known_institution '
                    'FROM "Author" au '
                    'JOIN "Author_Article" aa ON au.author_id = aa.author_id '
                    'JOIN "Article" a ON aa.article_id = a.article_id '
                    'JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id '
                    f'WHERE pas.project_id = {project_id} '
                    'GROUP BY au.author_id, au.display_name, au.h_index, au.last_known_institution '
                    'ORDER BY total_citations DESC LIMIT 5;'
                )
            return (
                'SELECT au.display_name, COALESCE(SUM(a.citation_count), 0) AS total_citations, COUNT(DISTINCT a.article_id) AS paper_count, au.h_index, au.last_known_institution '
                'FROM "Author" au '
                'JOIN "Author_Article" aa ON au.author_id = aa.author_id '
                'JOIN "Article" a ON aa.article_id = a.article_id '
                'GROUP BY au.author_id, au.display_name, au.h_index, au.last_known_institution '
                'ORDER BY total_citations DESC LIMIT 5;'
            )
        elif "tạp chí" in lower or "journal" in lower:
            if project_id:
                return (
                    'SELECT j.display_name, COUNT(DISTINCT a.article_id) AS paper_count '
                    'FROM "Journal" j '
                    'JOIN "Volume" v ON j.journal_id = v.journal_id '
                    'JOIN "Issue" i ON v.volume_id = i.volume_id '
                    'JOIN "Article" a ON i.issue_id = a.issue_id '
                    'JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id '
                    f'WHERE pas.project_id = {project_id} '
                    'GROUP BY j.journal_id, j.display_name '
                    'ORDER BY paper_count DESC LIMIT 5;'
                )
            return 'SELECT j.display_name, COUNT(DISTINCT a.article_id) AS paper_count FROM "Journal" j JOIN "Volume" v ON j.journal_id = v.journal_id JOIN "Issue" i ON v.volume_id = i.volume_id JOIN "Article" a ON i.issue_id = a.issue_id GROUP BY j.journal_id, j.display_name ORDER BY paper_count DESC LIMIT 5;'
        elif "chủ đề" in lower or "lĩnh vực" in lower or "topic" in lower:
            if project_id:
                return (
                    'SELECT t.display_name, COUNT(DISTINCT a.article_id) AS paper_count '
                    'FROM "Topic" t '
                    'JOIN "Article" a ON t.topic_id = a.primary_topic '
                    'JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id '
                    f'WHERE pas.project_id = {project_id} '
                    'GROUP BY t.topic_id, t.display_name '
                    'ORDER BY paper_count DESC LIMIT 5;'
                )
            return 'SELECT t.display_name, COUNT(DISTINCT a.article_id) AS paper_count FROM "Topic" t JOIN "Article" a ON t.topic_id = a.primary_topic GROUP BY t.topic_id, t.display_name ORDER BY paper_count DESC LIMIT 5;'
        elif "từ khóa" in lower or "keyword" in lower:
            if project_id:
                return (
                    'SELECT k.display_name, COUNT(DISTINCT ka.article_id) AS paper_count '
                    'FROM "Keyword" k '
                    'JOIN "Keyword_Article" ka ON k.keyword_id = ka.keyword_id '
                    'JOIN "Project_Article_Scope" pas ON ka.article_id = pas.article_id '
                    f'WHERE pas.project_id = {project_id} '
                    'GROUP BY k.keyword_id, k.display_name '
                    'ORDER BY paper_count DESC LIMIT 5;'
                )
            return 'SELECT k.display_name, COUNT(DISTINCT ka.article_id) AS paper_count FROM "Keyword" k JOIN "Keyword_Article" ka ON k.keyword_id = ka.keyword_id GROUP BY k.keyword_id, k.display_name ORDER BY paper_count DESC LIMIT 5;'
        # Specific country matching
        country_clause = None
        if any(c in lower for c in ["mỹ", "hoa kỳ", "united states", "usa"]):
            country_clause = "(z.name = 'United States' OR z.code = 'US')"
        elif any(c in lower for c in ["anh", "vương quốc anh", "united kingdom", "uk"]):
            country_clause = "(z.name = 'United Kingdom' OR z.code = 'GB')"
        elif any(c in lower for c in ["việt nam", "vietnam", "vn"]):
            country_clause = "(z.name ILIKE '%Viet%Nam%' OR z.code = 'VN')"
        elif any(c in lower for c in ["trung quốc", "china", "cn"]):
            country_clause = "(z.name = 'China' OR z.code = 'CN')"
        elif any(c in lower for c in ["nhật bản", "nhật", "japan", "jp"]):
            country_clause = "(z.name = 'Japan' OR z.code = 'JP')"
        elif any(c in lower for c in ["hàn quốc", "hàn", "korea", "kr"]):
            country_clause = "(z.name = 'South Korea' OR z.code = 'KR')"
        elif any(c in lower for c in ["đức", "germany", "de"]):
            country_clause = "(z.name = 'Germany' OR z.code = 'DE')"
        elif any(c in lower for c in ["pháp", "france", "fr"]):
            country_clause = "(z.name = 'France' OR z.code = 'FR')"
        elif any(c in lower for c in ["hà lan", "netherlands", "nl"]):
            country_clause = "(z.name = 'Netherlands' OR z.code = 'NL')"
        elif any(c in lower for c in ["thụy sĩ", "switzerland", "ch"]):
            country_clause = "(z.name = 'Switzerland' OR z.code = 'CH')"
        elif any(c in lower for c in ["tây ban nha", "spain", "es"]):
            country_clause = "(z.name = 'Spain' OR z.code = 'ES')"

        if country_clause:
            if project_id:
                return (
                    'SELECT COUNT(DISTINCT a.article_id) AS total_articles '
                    'FROM "Article" a '
                    'JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id '
                    'JOIN "Issue" i ON a.issue_id = i.issue_id '
                    'JOIN "Volume" v ON i.volume_id = v.volume_id '
                    'JOIN "Journal" j ON v.journal_id = j.journal_id '
                    'JOIN "Zone" z ON j.country = z.zone_id AND z.type = \'COUNTRY\' '
                    f'WHERE {country_clause} AND pas.project_id = {project_id};'
                )
            return (
                'SELECT COUNT(DISTINCT a.article_id) AS total_articles '
                'FROM "Article" a '
                'JOIN "Issue" i ON a.issue_id = i.issue_id '
                'JOIN "Volume" v ON i.volume_id = v.volume_id '
                'JOIN "Journal" j ON v.journal_id = j.journal_id '
                'JOIN "Zone" z ON j.country = z.zone_id AND z.type = \'COUNTRY\' '
                f'WHERE {country_clause};'
            )
        elif "quốc gia" in lower or "country" in lower or "nước" in lower:
            if project_id:
                return (
                    'SELECT z.name AS country_name, z.code AS country_code, COUNT(DISTINCT a.article_id) AS paper_count '
                    'FROM "Article" a '
                    'JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id '
                    'JOIN "Issue" i ON a.issue_id = i.issue_id '
                    'JOIN "Volume" v ON i.volume_id = v.volume_id '
                    'JOIN "Journal" j ON v.journal_id = j.journal_id '
                    'JOIN "Zone" z ON j.country = z.zone_id AND z.type = \'COUNTRY\' '
                    f'WHERE pas.project_id = {project_id} '
                    'GROUP BY z.name, z.code '
                    'ORDER BY paper_count DESC LIMIT 5;'
                )
            return 'SELECT z.name AS country_name, z.code AS country_code, COUNT(DISTINCT a.article_id) AS paper_count FROM "Article" a JOIN "Issue" i ON a.issue_id = i.issue_id JOIN "Volume" v ON i.volume_id = v.volume_id JOIN "Journal" j ON v.journal_id = j.journal_id JOIN "Zone" z ON j.country = z.zone_id AND z.type = \'COUNTRY\' GROUP BY z.name, z.code ORDER BY paper_count DESC LIMIT 5;'
        else:
            is_count = any(k in lower for k in [
                "bao nhiêu", "bao nhieu", "số lượng", "so luong", "tổng số", "tong so",
                "mấy", "may", "how many", "count", "total", "thống kê", "thong ke"
            ])
            years = sorted(list(set(int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", query))))
            year = years[0] if len(years) == 1 else None
            if is_count:
                if project_id:
                    year_filter = f' AND a.publication_year = {year}' if year else ""
                    return f'SELECT COUNT(DISTINCT a.article_id) FROM "Article" a JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id WHERE pas.project_id = {project_id}{year_filter};'
                year_clause = f' WHERE publication_year = {year}' if year else ""
                return f'SELECT COUNT(*) FROM "Article"{year_clause};'
            else:
                if project_id:
                    return (
                        'SELECT a.title, a.publication_year, COALESCE(a.citation_count, 0) AS citations, a.doi '
                        'FROM "Article" a '
                        'JOIN "Project_Article_Scope" pas ON a.article_id = pas.article_id '
                        f'WHERE pas.project_id = {project_id} '
                        'ORDER BY a.citation_count DESC NULLS LAST LIMIT 5;'
                    )
                return 'SELECT a.title, a.publication_year, COALESCE(a.citation_count, 0) AS citations, a.doi FROM "Article" a ORDER BY a.citation_count DESC NULLS LAST LIMIT 5;'

    def validate_and_sanitize(self, sql: str, project_id: Optional[int] = None) -> SQLValidationResult:
        """Enforces security guardrails: Read-only SELECT, anti-injection, and forced LIMIT."""
        if not sql or not sql.strip():
            return SQLValidationResult(is_valid=False, error_message="Empty SQL string")

        clean = sql.strip().rstrip(";")

        # Rule 1: Must start with SELECT
        if not re.match(r"^\s*SELECT\b", clean, re.IGNORECASE):
            return SQLValidationResult(
                is_valid=False,
                error_message="Only SELECT queries are allowed for read-only retrieval.",
            )

        # Rule 2: Forbidden destructive DDL/DML keywords
        for dis in self.DISALLOWED_KEYWORDS:
            if re.search(dis, clean, re.IGNORECASE):
                return SQLValidationResult(
                    is_valid=False,
                    error_message=f"Potentially destructive SQL command rejected: {dis}",
                )

        # Rule 3: Reject multi-statement queries (semicolon in middle)
        if ";" in clean:
            return SQLValidationResult(
                is_valid=False,
                error_message="Multi-statement SQL queries are strictly prohibited.",
            )

        # Rule 4: Enforce LIMIT for listing queries
        if not re.search(r"\bCOUNT\b", clean, re.IGNORECASE) and not re.search(r"\bLIMIT\s+\d+\b", clean, re.IGNORECASE):
            clean += " LIMIT 10"

        # Rule 5: Quote normalization for known tables if LLM generated unquoted table names
        table_names = [
            "Article", "Author", "Author_Article", "Journal", "Volume",
            "Issue", "Topic", "Keyword", "Keyword_Article", "Zone", "Project_Article_Scope",
        ]
        for tbl in table_names:
            clean = re.sub(rf'(?<!")\b{tbl}\b(?!")', f'"{tbl}"', clean)

        # Rule 6: Enforce project scope dynamically
        if project_id:
            # Overwrite any hallucinated or mismatched project_id with the active project_id
            clean = re.sub(r'\bpas\.project_id\s*=\s*\d+', f'pas.project_id = {project_id}', clean)
            clean = re.sub(r'\bproject_id\s*=\s*\d+', f'project_id = {project_id}', clean)

            # If Project_Article_Scope is missing entirely, inject it
            if "Project_Article_Scope" not in clean and "pas." not in clean:
                if re.search(r'\bWHERE\b', clean, re.IGNORECASE):
                    clean = re.sub(
                        r'\bWHERE\b',
                        f'JOIN "Project_Article_Scope" pas ON pas.article_id = a.article_id WHERE pas.project_id = {project_id} AND' if ' a.' in clean or ' "Article" a' in clean else f'WHERE article_id IN (SELECT pas.article_id FROM "Project_Article_Scope" pas WHERE pas.project_id = {project_id}) AND',
                        clean,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                elif re.search(r'\b(GROUP BY|ORDER BY|LIMIT)\b', clean, re.IGNORECASE):
                    clean = re.sub(
                        r'\b(GROUP BY|ORDER BY|LIMIT)\b',
                        rf'WHERE article_id IN (SELECT pas.article_id FROM "Project_Article_Scope" pas WHERE pas.project_id = {project_id}) \1',
                        clean,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                else:
                    clean += f' WHERE article_id IN (SELECT pas.article_id FROM "Project_Article_Scope" pas WHERE pas.project_id = {project_id})'
        else:
            # System-wide scope: Remove any leftover project_id constraints
            clean = re.sub(r'\s+AND\s+pas\.project_id\s*=\s*\d+', '', clean)
            clean = re.sub(r'\s+WHERE\s+pas\.project_id\s*=\s*\d+\s+AND\s+', ' WHERE ', clean)
            clean = re.sub(r'\s+WHERE\s+pas\.project_id\s*=\s*\d+', '', clean)

        # Rule 7: Map Vietnamese country names in WHERE clauses to Zone table English name/code
        country_aliases = [
            (r"(?:z\.name|country_zone\.name)\s*(=|ILIKE|LIKE)\s*['\"]Mỹ['\"]", "(z.name = 'United States' OR z.code = 'US')"),
            (r"(?:z\.name|country_zone\.name)\s*(=|ILIKE|LIKE)\s*['\"]Hoa Kỳ['\"]", "(z.name = 'United States' OR z.code = 'US')"),
            (r"(?:z\.name|country_zone\.name)\s*(=|ILIKE|LIKE)\s*['\"]Anh['\"]", "(z.name = 'United Kingdom' OR z.code = 'GB')"),
            (r"(?:z\.name|country_zone\.name)\s*(=|ILIKE|LIKE)\s*['\"]Vương quốc Anh['\"]", "(z.name = 'United Kingdom' OR z.code = 'GB')"),
            (r"(?:z\.name|country_zone\.name)\s*(=|ILIKE|LIKE)\s*['\"]Việt Nam['\"]", "(z.name ILIKE '%Viet%Nam%' OR z.code = 'VN')"),
            (r"(?:z\.name|country_zone\.name)\s*(=|ILIKE|LIKE)\s*['\"]Trung Quốc['\"]", "(z.name = 'China' OR z.code = 'CN')"),
            (r"(?:z\.name|country_zone\.name)\s*(=|ILIKE|LIKE)\s*['\"]Nhật Bản['\"]", "(z.name = 'Japan' OR z.code = 'JP')"),
            (r"(?:z\.name|country_zone\.name)\s*(=|ILIKE|LIKE)\s*['\"]Hàn Quốc['\"]", "(z.name = 'South Korea' OR z.code = 'KR')"),
            (r"(?:z\.name|country_zone\.name)\s*(=|ILIKE|LIKE)\s*['\"]Đức['\"]", "(z.name = 'Germany' OR z.code = 'DE')"),
            (r"(?:z\.name|country_zone\.name)\s*(=|ILIKE|LIKE)\s*['\"]Pháp['\"]", "(z.name = 'France' OR z.code = 'FR')"),
            (r"(?:z\.name|country_zone\.name)\s*(=|ILIKE|LIKE)\s*['\"]Hà Lan['\"]", "(z.name = 'Netherlands' OR z.code = 'NL')"),
            (r"(?:z\.name|country_zone\.name)\s*(=|ILIKE|LIKE)\s*['\"]Thụy Sĩ['\"]", "(z.name = 'Switzerland' OR z.code = 'CH')"),
            (r"(?:z\.name|country_zone\.name)\s*(=|ILIKE|LIKE)\s*['\"]Tây Ban Nha['\"]", "(z.name = 'Spain' OR z.code = 'ES')"),
        ]
        for pattern, replacement in country_aliases:
            clean = re.sub(pattern, replacement, clean, flags=re.IGNORECASE)

        return SQLValidationResult(is_valid=True, sanitized_sql=clean)

    def execute_and_format(self, query: str, project_id: Optional[int] = None) -> RetrievedChunk:
        """Generates SQL via LLM, validates, executes with 1-shot self-correction, and formats as RetrievedChunk."""
        t0 = time.perf_counter()

        # Step 1: Generate SQL from natural language query
        raw_sql = self.generate_sql(query, project_id)
        validation = self.validate_and_sanitize(raw_sql, project_id)

        if not validation.is_valid:
            logger.warning(f"[TextToSQL] Validation failed: {validation.error_message}. Falling back.")
            raw_sql = self._generate_fallback_sql(query, project_id)
            validation = self.validate_and_sanitize(raw_sql, project_id)

        final_sql = validation.sanitized_sql or raw_sql
        scope_title = f"[Thống kê trong phạm vi Đề tài Dự án #{project_id}]" if project_id else "[Thống kê cơ sở dữ liệu ResearchPulse]"

        # Step 2: Execute SQL on PostgreSQL
        exec_res = self._execute_query(final_sql)

        # Step 3: 1-Shot Self-Correction if error occurred
        if not exec_res.success and exec_res.error:
            logger.info(f"[TextToSQL] SQL error detected: {exec_res.error}. Attempting 1-shot self-correction...")
            corrected_sql = self._self_correct_sql(query, final_sql, exec_res.error, project_id)
            val_corr = self.validate_and_sanitize(corrected_sql, project_id)
            if val_corr.is_valid and val_corr.sanitized_sql:
                exec_res = self._execute_query(val_corr.sanitized_sql)
                if exec_res.success:
                    final_sql = val_corr.sanitized_sql

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Step 4: Build rich human-readable markdown content
        if exec_res.success and exec_res.rows:
            formatted_table = self._build_markdown_output(scope_title, query, exec_res)
        elif exec_res.success and not exec_res.rows:
            formatted_table = (
                f"{scope_title}\n"
                f"Truy vấn SQL thực thi thành công nhưng không tìm thấy dữ liệu phù hợp với yêu cầu: '{query}'.\n"
                f"SQL: `{final_sql}`"
            )
        else:
            # Fallback if both tries failed
            fallback_sql = self._generate_fallback_sql(query, project_id)
            exec_res = self._execute_query(fallback_sql)
            if exec_res.success and exec_res.rows:
                formatted_table = self._build_markdown_output(scope_title, query, exec_res)
            else:
                formatted_table = (
                    f"{scope_title}\n"
                    f"Không thể thực thi truy vấn thống kê dữ liệu cho câu hỏi: '{query}'."
                )

        return RetrievedChunk(
            chunk_id=f"text_to_sql_{int(time.time())}",
            document_id="doc_text_to_sql_result",
            content=formatted_table,
            score=1.0,
            rerank_score=1.0,
            metadata={
                "source": "postgresql_sql",
                "type": "Aggregation",
                "sql": final_sql,
                "project_id": project_id,
                "latency_ms": latency_ms,
            },
        )

    def _execute_query(self, sql: str) -> SQLExecutionResult:
        """Executes SQL against PostgreSQL and returns structured result."""
        t0 = time.perf_counter()
        conn = self._get_postgres_connection()
        if not conn:
            return SQLExecutionResult(
                sql=sql,
                success=False,
                error="Could not establish connection to PostgreSQL.",
                execution_time_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                cols = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall() if cur.description else []
                conn.commit()
                return SQLExecutionResult(
                    sql=sql,
                    columns=cols,
                    rows=[list(r) for r in rows],
                    row_count=len(rows),
                    success=True,
                    execution_time_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
        except Exception as e:
            if conn:
                conn.rollback()
            return SQLExecutionResult(
                sql=sql,
                success=False,
                error=str(e),
                execution_time_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        finally:
            if conn:
                conn.close()

    def _self_correct_sql(self, query: str, faulty_sql: str, error_msg: str, project_id: Optional[int]) -> str:
        """Passes SQL error back to LLM to self-correct."""
        prompt = (
            f"Câu lệnh SQL sau đây bị lỗi khi thực thi trên PostgreSQL:\n"
            f"SQL lỗi: {faulty_sql}\n"
            f"Thông báo lỗi: {error_msg}\n"
            f"Câu hỏi gốc: {query}\n"
            f"Project ID: {project_id}\n"
            f"Hãy sửa lại câu SQL trên để chạy chính xác. Trả về DUY NHẤT câu SQL mới, không giải thích:"
        )
        if self.settings.LLM_PROVIDER == "ollama" or getattr(self.settings, "OLLAMA_BASE_URL", None):
            try:
                ollama_url = f"{self.settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
                payload = {
                    "model": self.settings.OLLAMA_MODEL or "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 256},
                }
                req = urllib.request.Request(
                    ollama_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return self._clean_sql(data.get("response", ""))
            except Exception:
                pass
        return self._generate_fallback_sql(query, project_id)

    def _build_markdown_output(self, scope_title: str, query: str, exec_res: SQLExecutionResult) -> str:
        """Formats SQL execution results into structured Markdown for generation context."""
        cols = exec_res.columns
        rows = exec_res.rows

        # Case A: Single Scalar Count (e.g. SELECT COUNT(*))
        if len(cols) == 1 and len(rows) == 1 and isinstance(rows[0][0], (int, float)):
            metric_val = rows[0][0]
            col_name = cols[0].replace("_", " ").title()
            if any(k in query.lower() for k in ["tác giả", "author"]):
                unit = "tác giả (nhà khoa học)"
            elif any(k in query.lower() for k in ["tạp chí", "journal"]):
                unit = "tạp chí khoa học"
            elif any(k in query.lower() for k in ["chủ đề", "topic", "lĩnh vực"]):
                unit = "chủ đề nghiên cứu"
            elif any(k in query.lower() for k in ["từ khóa", "keyword"]):
                unit = "từ khóa học thuật"
            elif any(k in query.lower() for k in ["quốc gia", "country", "nước"]):
                unit = "quốc gia / vùng lãnh thổ"
            else:
                unit = "bài báo khoa học"
            return (
                f"{scope_title}\n"
                f"Kết quả truy vấn dữ liệu cho câu hỏi: '{query}':\n"
                f"👉 **{col_name}**: **{metric_val:,} {unit}**\n\n"
                f"*(Nguồn: Cơ sở dữ liệu PostgreSQL thực thi qua Dynamic Text-to-SQL)*"
            )

        # Case A2: Single Row with Multiple Metrics (e.g. Total Authors + Total Articles)
        if len(rows) == 1 and all(isinstance(v, (int, float)) for v in rows[0] if v is not None):
            lines = [f"{scope_title}", f"Kết quả thống kê tổng hợp từ cơ sở dữ liệu cho câu hỏi: '{query}':\n"]
            for col, val in zip(cols, rows[0]):
                if val is None:
                    continue
                col_lower = col.lower()
                friendly_col = col.replace("_", " ").title()
                if "author" in col_lower or "tác giả" in col_lower:
                    unit = "tác giả (nhà khoa học)"
                elif "article" in col_lower or "paper" in col_lower or "bài báo" in col_lower:
                    unit = "bài báo khoa học"
                elif "journal" in col_lower or "tạp chí" in col_lower:
                    unit = "tạp chí"
                elif "citation" in col_lower or "trích dẫn" in col_lower:
                    unit = "lượt trích dẫn"
                else:
                    unit = ""
                unit_str = f" {unit}" if unit else ""
                lines.append(f"- 👉 **{friendly_col}**: **{val:,}**{unit_str}")
            lines.append("\n*(Nguồn: Cơ sở dữ liệu PostgreSQL thực thi qua Dynamic Text-to-SQL)*")
            return "\n".join(lines)

        # Case B: Multi-row or Multi-column (Rankings, Top Listings, Breakdown)
        lines = [f"{scope_title}", f"Kết quả thống kê và xếp hạng từ cơ sở dữ liệu:\n"]

        # If it's a ranking, formulate top 1 highlighted
        if rows:
            top_1_name = rows[0][0]
            top_1_val = rows[0][1] if len(rows[0]) > 1 else ""
            if isinstance(top_1_val, (int, float)):
                top_1_val_str = f"**{top_1_val:,}**"
            else:
                top_1_val_str = f"**{top_1_val}**" if top_1_val else ""

            lines.append(f"👉 **Dẫn đầu (Hạng 1)**: **{top_1_name}** {top_1_val_str}\n")

        # Build Markdown list of rows
        for rank, row in enumerate(rows[:10], 1):
            row_parts = []
            for col_idx, (col, val) in enumerate(zip(cols, row)):
                if val is None:
                    continue
                if isinstance(val, (int, float)):
                    val_str = f"{val:,}"
                else:
                    val_str = str(val)

                if col_idx == 0:
                    row_parts.append(f"**{val_str}**")
                else:
                    friendly_col = col.replace("_", " ").title()
                    row_parts.append(f"{friendly_col}: {val_str}")

            lines.append(f"- Hạng {rank}: " + " | ".join(row_parts))

        return "\n".join(lines)
