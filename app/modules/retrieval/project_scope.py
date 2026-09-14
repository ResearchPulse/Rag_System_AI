import logging
from dataclasses import dataclass, field
from typing import List, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ProjectScopeMetadata:
    """Metadata representing the academic scope of a user's research project."""
    project_id: int
    title: str = ""
    description: str = ""
    subject_area_id: Optional[int] = None
    subject_area_name: str = ""
    subject_category_ids: List[int] = field(default_factory=list)
    keyword_ids: List[int] = field(default_factory=list)
    keyword_names: List[str] = field(default_factory=list)
    article_count: int = 0

    def to_context_summary(self) -> str:
        """Formats project scope into a crisp prompt context for LLM generation."""
        kw_str = ", ".join(self.keyword_names[:8]) if self.keyword_names else "Không có từ khóa cố định"
        subj_str = self.subject_area_name or "Đa ngành"
        desc_str = f" • *Mục tiêu*: {self.description[:180]}..." if self.description else ""
        return (
            f"### 🎯 Phạm vi Đề tài Nghiên cứu của Dự án (Project Scope #{self.project_id})\n"
            f"- **Tên Đề tài / Dự án**: {self.title or f'Dự án #{self.project_id}'}\n"
            f"- **Lĩnh vực Chuyên môn**: {subj_str}\n"
            f"- **Từ khóa Trọng tâm**: {kw_str}\n"
            f"- **Quy mô Kho Tri thức Thuộc Phạm vi**: {self.article_count:,} bài báo khoa học đã được chọn lọc{desc_str}\n"
            f"*Lưu ý: Mọi phân tích, dữ liệu thống kê, bài báo và mạng lưới tác giả dưới đây đều được giới hạn nghiêm ngặt "
            f"trong phạm vi chuyên môn của Đề tài này.*"
        )


class ProjectScopeService:
    """Service for resolving, ensuring, and querying project-scoped academic data."""

    _cached_scopes = {}
    _db_available = True
    _last_db_check = 0.0

    def __init__(self) -> None:
        self.settings = get_settings()

    def _get_connection(self):
        """Attempts connection to PostgreSQL with fallback hosts."""
        import time
        now = time.time()
        if not ProjectScopeService._db_available and (now - ProjectScopeService._last_db_check < 15.0):
            return None

        import psycopg2

        # 1. Direct POSTGRES_URL
        if getattr(self.settings, "POSTGRES_URL", None):
            try:
                conn = psycopg2.connect(self.settings.POSTGRES_URL, connect_timeout=1)
                ProjectScopeService._db_available = True
                return conn
            except Exception:
                pass

        # 2. Configured host
        primary_host = "127.0.0.1" if self.settings.POSTGRES_HOST in ("localhost", "127.0.0.1") else self.settings.POSTGRES_HOST
        dsn = (
            f"postgresql://{self.settings.POSTGRES_USER}:{self.settings.POSTGRES_PASSWORD}"
            f"@{primary_host}:{self.settings.POSTGRES_PORT}/{self.settings.POSTGRES_DB}"
        )
        try:
            conn = psycopg2.connect(dsn, connect_timeout=1)
            ProjectScopeService._db_available = True
            return conn
        except Exception:
            pass

        # 3. Remote fallback
        if primary_host != "100.121.61.95":
            try:
                dsn_remote = (
                    f"postgresql://{self.settings.POSTGRES_USER}:{self.settings.POSTGRES_PASSWORD}"
                    f"@100.121.61.95:{self.settings.POSTGRES_PORT}/{self.settings.POSTGRES_DB}"
                )
                conn = psycopg2.connect(dsn_remote, connect_timeout=1)
                ProjectScopeService._db_available = True
                return conn
            except Exception:
                pass

        ProjectScopeService._db_available = False
        ProjectScopeService._last_db_check = now
        return None

    def get_project_metadata(self, project_id: int) -> Optional[ProjectScopeMetadata]:
        """Loads project definition: subject area, categories, keywords, and scope count."""
        if not project_id:
            return None

        # Return cached metadata if present
        if project_id in self._cached_scopes:
            return self._cached_scopes[project_id]

        conn = self._get_connection()
        if not conn:
            # Fallback mock scope when database is offline
            mock_scope = ProjectScopeMetadata(
                project_id=project_id,
                title=f"Nghiên cứu Khoa học Dự án #{project_id}",
                description="Đề tài theo dõi xu hướng công bố khoa học trong phạm vi dự án.",
                subject_area_name="Khoa học Máy tính & Trí tuệ Nhân tạo",
                keyword_names=["Artificial Intelligence", "Machine Learning", "Graph RAG"],
                article_count=150,
            )
            return mock_scope

        try:
            with conn:
                with conn.cursor() as cur:
                    # 1. Fetch Project & Subject Area
                    cur.execute("""
                        SELECT p.project_id, p.title, COALESCE(p.description, ''), sa.subject_area_id, sa.display_name
                        FROM "Project" p
                        LEFT JOIN "Subject_Area" sa ON p.subject_area = sa.subject_area_id
                        WHERE p.project_id = %s;
                    """, (project_id,))
                    p_row = cur.fetchone()
                    if not p_row:
                        return None

                    p_id, title, desc, sa_id, sa_name = p_row

                    # 2. Fetch Subject Categories
                    cat_ids = []
                    if sa_id:
                        cur.execute("""
                            SELECT subject_category_id
                            FROM "Subject_Category"
                            WHERE subject_area_id = %s AND COALESCE(is_deleted, false) = false;
                        """, (sa_id,))
                        cat_ids = [r[0] for r in cur.fetchall()]

                    # 3. Fetch Keywords
                    cur.execute("""
                        SELECT k.keyword_id, k.display_name
                        FROM "Project_Keyword" pk
                        JOIN "Keyword" k ON pk.keyword_id = k.keyword_id
                        WHERE pk.project_id = %s;
                    """, (project_id,))
                    kw_rows = cur.fetchall()
                    kw_ids = [r[0] for r in kw_rows]
                    kw_names = [r[1] for r in kw_rows]

                    # 4. Check / Ensure Project_Article_Scope count
                    cur.execute("""
                        SELECT COUNT(*) FROM "Project_Article_Scope" WHERE project_id = %s;
                    """, (project_id,))
                    art_count = cur.fetchone()[0]

                    # If scope count is 0, attempt auto-population on-demand
                    if art_count == 0:
                        art_count = self._populate_scope_if_needed(cur, project_id, cat_ids, kw_ids)

                    scope = ProjectScopeMetadata(
                        project_id=p_id,
                        title=title or f"Dự án #{project_id}",
                        description=desc or "",
                        subject_area_id=sa_id,
                        subject_area_name=sa_name or "",
                        subject_category_ids=cat_ids,
                        keyword_ids=kw_ids,
                        keyword_names=kw_names,
                        article_count=art_count,
                    )
                    self._cached_scopes[project_id] = scope
                    return scope
        except Exception as e:
            logger.warning("Error fetching project scope metadata for #%s: %s", project_id, e)
            return None
        finally:
            conn.close()

    def _populate_scope_if_needed(
        self,
        cur,
        project_id: int,
        cat_ids: List[int],
        kw_ids: List[int],
    ) -> int:
        """On-demand population of Project_Article_Scope if currently empty."""
        try:
            if cat_ids:
                cur.execute("""
                    INSERT INTO "Project_Article_Scope" (project_id, article_id, publication_year)
                    SELECT DISTINCT %s, a.article_id, a.publication_year
                    FROM "Article" a
                    WHERE a.primary_topic IN (
                        SELECT topic_id FROM "Topic" WHERE subject_category_id = ANY(%s)
                    )
                    ON CONFLICT DO NOTHING;
                """, (project_id, cat_ids))

                cur.execute("""
                    INSERT INTO "Project_Article_Scope" (project_id, article_id, publication_year)
                    SELECT DISTINCT %s, a.article_id, a.publication_year
                    FROM "Sub_Topic" st
                    JOIN "Topic" sub_topic ON st.topic_id = sub_topic.topic_id
                    JOIN "Article" a ON st.article_id = a.article_id
                    WHERE sub_topic.subject_category_id = ANY(%s)
                    ON CONFLICT DO NOTHING;
                """, (project_id, cat_ids))

            if kw_ids:
                cur.execute("""
                    INSERT INTO "Project_Article_Scope" (project_id, article_id, publication_year)
                    SELECT DISTINCT %s, a.article_id, a.publication_year
                    FROM "Keyword_Article" ka
                    JOIN "Article" a ON ka.article_id = a.article_id
                    WHERE ka.keyword_id = ANY(%s)
                    ON CONFLICT DO NOTHING;
                """, (project_id, kw_ids))

            cur.execute("""
                SELECT COUNT(*) FROM "Project_Article_Scope" WHERE project_id = %s;
            """, (project_id,))
            return cur.fetchone()[0]
        except Exception as e:
            logger.debug("Scope population skipped or failed: %s", e)
            return 0

    def get_scoped_authors(self, project_id: int, limit: int = 15) -> List[str]:
        """Retrieves notable authors who authored papers within this project's scope."""
        conn = self._get_connection()
        if not conn:
            return []

        try:
            with conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT au.author_name, COUNT(pas.article_id) as paper_cnt
                        FROM "Author" au
                        JOIN "Author_Article" aa ON au.author_id = aa.author_id
                        JOIN "Project_Article_Scope" pas ON aa.article_id = pas.article_id
                        WHERE pas.project_id = %s
                        GROUP BY au.author_name
                        ORDER BY paper_cnt DESC, MAX(au.citation_count) DESC
                        LIMIT %s;
                    """
                    cur.execute(query, (project_id, limit))
                    return [r[0] for r in cur.fetchall() if r[0]]
        except Exception as e:
            logger.debug("Error getting scoped authors for project #%s: %s", project_id, e)
            return []
        finally:
            conn.close()
