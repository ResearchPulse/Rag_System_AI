import logging
from datetime import datetime
from typing import List, Optional, Tuple

from app.core.config import get_settings
from app.modules.chat_history.schemas import (
    ChatMessageCreate,
    ChatMessageItem,
    ChatMessageRole,
    ChatMessageStatus,
)

logger = logging.getLogger(__name__)


class ChatHistoryRepository:
    """Repository handling database operations for Chat Messages in PostgreSQL."""

    _shared_in_memory_messages: List[dict] = []
    _shared_in_memory_seq: int = 1
    _working_dsn: Optional[str] = None
    _last_db_check: float = 0.0
    _db_available: Optional[bool] = None

    def __init__(self) -> None:
        self.settings = get_settings()
        self._table_initialized = False

    @property
    def _in_memory_messages(self) -> List[dict]:
        return ChatHistoryRepository._shared_in_memory_messages

    @_in_memory_messages.setter
    def _in_memory_messages(self, val: List[dict]) -> None:
        ChatHistoryRepository._shared_in_memory_messages = val

    @property
    def _in_memory_seq(self) -> int:
        return ChatHistoryRepository._shared_in_memory_seq

    @_in_memory_seq.setter
    def _in_memory_seq(self, val: int) -> None:
        ChatHistoryRepository._shared_in_memory_seq = val

    def get_connection(self):
        """Attempts connection to PostgreSQL with fallback hosts and caching."""
        import psycopg2
        import time

        now = time.time()
        # If checked within last 15 seconds and DB was unavailable, return None immediately
        if ChatHistoryRepository._db_available is False and (now - ChatHistoryRepository._last_db_check) < 15.0:
            return None

        # 1. If we have a known working DSN, try it first
        if ChatHistoryRepository._working_dsn:
            try:
                conn = psycopg2.connect(ChatHistoryRepository._working_dsn, connect_timeout=1)
                ChatHistoryRepository._db_available = True
                ChatHistoryRepository._last_db_check = now
                return conn
            except Exception:
                ChatHistoryRepository._working_dsn = None

        candidate_dsns = []
        if getattr(self.settings, "POSTGRES_URL", None):
            candidate_dsns.append(self.settings.POSTGRES_URL)

        primary_host = "127.0.0.1" if self.settings.POSTGRES_HOST in ("localhost", "127.0.0.1") else self.settings.POSTGRES_HOST
        candidate_dsns.append(
            f"postgresql://{self.settings.POSTGRES_USER}:{self.settings.POSTGRES_PASSWORD}"
            f"@{primary_host}:{self.settings.POSTGRES_PORT}/{self.settings.POSTGRES_DB}"
        )
        if "100.121.61.95" != primary_host:
            candidate_dsns.append(
                f"postgresql://{self.settings.POSTGRES_USER}:{self.settings.POSTGRES_PASSWORD}"
                f"@100.121.61.95:{self.settings.POSTGRES_PORT}/{self.settings.POSTGRES_DB}"
            )

        for dsn in candidate_dsns:
            try:
                conn = psycopg2.connect(dsn, connect_timeout=1)
                ChatHistoryRepository._working_dsn = dsn
                ChatHistoryRepository._db_available = True
                ChatHistoryRepository._last_db_check = now
                return conn
            except Exception:
                continue

        ChatHistoryRepository._db_available = False
        ChatHistoryRepository._last_db_check = now
        return None


    def ensure_tables_exist(self) -> bool:
        """Idempotently ensures enum types and Project_Chat_Message table exist."""
        if self._table_initialized:
            return True

        conn = self.get_connection()
        if not conn:
            logger.warning("PostgreSQL connection not available; using in-memory store for chat messages.")
            return False

        try:
            with conn:
                with conn.cursor() as cur:
                    # 1. Create Enums if they don't exist
                    cur.execute("""
                        DO $$
                        BEGIN
                            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'message_role') THEN
                                CREATE TYPE message_role AS ENUM ('USER', 'ASSISTANT', 'SYSTEM');
                            END IF;
                            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'message_status') THEN
                                CREATE TYPE message_status AS ENUM ('PENDING', 'COMPLETED', 'ERROR');
                            END IF;
                        END$$;
                    """)

                    # 2. Create Project_Chat_Message table if not exists
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS "Project_Chat_Message" (
                            message_id BIGSERIAL PRIMARY KEY,
                            project_id BIGINT,
                            user_id UUID NOT NULL,
                            role message_role NOT NULL DEFAULT 'USER',
                            content TEXT NOT NULL,
                            model TEXT,
                            prompt_tokens INTEGER DEFAULT 0,
                            completion_tokens INTEGER DEFAULT 0,
                            total_tokens INTEGER DEFAULT 0,
                            latency_ms INTEGER,
                            status message_status NOT NULL DEFAULT 'COMPLETED',
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                        );
                        CREATE INDEX IF NOT EXISTS idx_chat_msg_proj_user 
                            ON "Project_Chat_Message" (project_id, user_id, created_at DESC);
                        CREATE INDEX IF NOT EXISTS idx_chat_msg_user 
                            ON "Project_Chat_Message" (user_id, created_at DESC);
                    """)
            self._table_initialized = True
            logger.info("Project_Chat_Message table and indices verified successfully.")
            return True
        except Exception as e:
            logger.warning("Error initializing Project_Chat_Message table: %s", e)
            return False
        finally:
            conn.close()

    def create_message(self, data: ChatMessageCreate) -> ChatMessageItem:
        """Persists a new chat message into DB or fallback memory store."""
        self.ensure_tables_exist()
        conn = self.get_connection()

        if conn:
            try:
                with conn:
                    with conn.cursor() as cur:
                        query = """
                            INSERT INTO "Project_Chat_Message" (
                                project_id,
                                user_id,
                                role,
                                content,
                                model,
                                prompt_tokens,
                                completion_tokens,
                                total_tokens,
                                latency_ms,
                                status
                            )
                            VALUES (%s, %s::uuid, %s::message_role, %s, %s, %s, %s, %s, %s, %s::message_status)
                            RETURNING message_id, project_id, user_id::text, role::text, content, model, 
                                      prompt_tokens, completion_tokens, total_tokens, latency_ms, status::text, created_at;
                        """
                        cur.execute(
                            query,
                            (
                                data.project_id,
                                data.user_id,
                                data.role.value,
                                data.content,
                                data.model,
                                data.prompt_tokens,
                                data.completion_tokens,
                                data.total_tokens,
                                data.latency_ms,
                                data.status.value,
                            ),
                        )
                        row = cur.fetchone()
                        return ChatMessageItem(
                            message_id=row[0],
                            project_id=row[1],
                            user_id=row[2],
                            role=row[3],
                            content=row[4],
                            model=row[5],
                            prompt_tokens=row[6],
                            completion_tokens=row[7],
                            total_tokens=row[8],
                            latency_ms=row[9],
                            status=row[10],
                            created_at=row[11],
                        )
            except Exception as e:
                logger.error("Database insert error in create_message: %s; saving to in-memory fallback.", e)
            finally:
                conn.close()

        # In-memory fallback
        msg_id = self._in_memory_seq
        self._in_memory_seq += 1
        now = datetime.now()
        item = ChatMessageItem(
            message_id=msg_id,
            project_id=data.project_id,
            user_id=data.user_id,
            role=data.role.value,
            content=data.content,
            model=data.model,
            prompt_tokens=data.prompt_tokens,
            completion_tokens=data.completion_tokens,
            total_tokens=data.total_tokens,
            latency_ms=data.latency_ms,
            status=data.status.value,
            created_at=now,
        )
        self._in_memory_messages.append(item.model_dump())
        return item

    def get_messages(
        self,
        project_id: Optional[int] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        order: str = "asc",
    ) -> Tuple[List[ChatMessageItem], int]:
        """Retrieves paginated messages filtered by project_id and/or user_id."""
        self.ensure_tables_exist()
        order_dir = "DESC" if order.lower() == "desc" else "ASC"
        conn = self.get_connection()

        if conn:
            try:
                with conn:
                    with conn.cursor() as cur:
                        where_clauses = []
                        params = []

                        if project_id is not None:
                            where_clauses.append("project_id = %s")
                            params.append(project_id)

                        if user_id is not None:
                            where_clauses.append("user_id = %s::uuid")
                            params.append(user_id)

                        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

                        # 1. Total count
                        count_sql = f'SELECT COUNT(*) FROM "Project_Chat_Message" {where_sql};'
                        cur.execute(count_sql, tuple(params))
                        total_count = cur.fetchone()[0]

                        # 2. Paginated rows
                        query_sql = f"""
                            SELECT message_id, project_id, user_id::text, role::text, content, model, 
                                   prompt_tokens, completion_tokens, total_tokens, latency_ms, status::text, created_at
                            FROM "Project_Chat_Message"
                            {where_sql}
                            ORDER BY created_at {order_dir}, message_id {order_dir}
                            LIMIT %s OFFSET %s;
                        """
                        cur.execute(query_sql, tuple(params + [limit, offset]))
                        rows = cur.fetchall()

                        items = [
                            ChatMessageItem(
                                message_id=r[0],
                                project_id=r[1],
                                user_id=r[2],
                                role=r[3],
                                content=r[4],
                                model=r[5],
                                prompt_tokens=r[6],
                                completion_tokens=r[7],
                                total_tokens=r[8],
                                latency_ms=r[9],
                                status=r[10],
                                created_at=r[11],
                            )
                            for r in rows
                        ]
                        return items, total_count
            except Exception as e:
                logger.error("Database query error in get_messages: %s; falling back to in-memory.", e)
            finally:
                conn.close()

        # In-memory fallback filtering
        filtered = self._in_memory_messages
        if project_id is not None:
            filtered = [m for m in filtered if m.get("project_id") == project_id]
        if user_id is not None:
            filtered = [m for m in filtered if m.get("user_id") == user_id]

        total = len(filtered)
        is_reverse = (order.lower() == "desc")
        sorted_list = sorted(filtered, key=lambda x: (x.get("created_at") or datetime.min, x.get("message_id")), reverse=is_reverse)
        paged = sorted_list[offset : offset + limit]
        return [ChatMessageItem(**m) for m in paged], total

    def get_message_by_id(
        self,
        message_id: int,
        project_id: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> Optional[ChatMessageItem]:
        """Retrieves a single message by ID."""
        self.ensure_tables_exist()
        conn = self.get_connection()

        if conn:
            try:
                with conn:
                    with conn.cursor() as cur:
                        where_clauses = ["message_id = %s"]
                        params = [message_id]
                        if project_id is not None:
                            where_clauses.append("project_id = %s")
                            params.append(project_id)
                        if user_id is not None:
                            where_clauses.append("user_id = %s::uuid")
                            params.append(user_id)

                        query = f"""
                            SELECT message_id, project_id, user_id::text, role::text, content, model, 
                                   prompt_tokens, completion_tokens, total_tokens, latency_ms, status::text, created_at
                            FROM "Project_Chat_Message"
                            WHERE {' AND '.join(where_clauses)}
                            LIMIT 1;
                        """
                        cur.execute(query, tuple(params))
                        r = cur.fetchone()
                        if r:
                            return ChatMessageItem(
                                message_id=r[0],
                                project_id=r[1],
                                user_id=r[2],
                                role=r[3],
                                content=r[4],
                                model=r[5],
                                prompt_tokens=r[6],
                                completion_tokens=r[7],
                                total_tokens=r[8],
                                latency_ms=r[9],
                                status=r[10],
                                created_at=r[11],
                            )
                        return None
            except Exception as e:
                logger.error("Error in get_message_by_id: %s", e)
            finally:
                conn.close()

        # In-memory lookup
        for m in self._in_memory_messages:
            if m.get("message_id") == message_id:
                if project_id is not None and m.get("project_id") != project_id:
                    continue
                if user_id is not None and m.get("user_id") != user_id:
                    continue
                return ChatMessageItem(**m)
        return None

    def delete_message(
        self,
        message_id: int,
        project_id: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """Deletes a specific message."""
        self.ensure_tables_exist()
        conn = self.get_connection()

        if conn:
            try:
                with conn:
                    with conn.cursor() as cur:
                        where_clauses = ["message_id = %s"]
                        params = [message_id]
                        if project_id is not None:
                            where_clauses.append("project_id = %s")
                            params.append(project_id)
                        if user_id is not None:
                            where_clauses.append("user_id = %s::uuid")
                            params.append(user_id)

                        sql = f'DELETE FROM "Project_Chat_Message" WHERE {" AND ".join(where_clauses)};'
                        cur.execute(sql, tuple(params))
                        return cur.rowcount
            except Exception as e:
                logger.error("Error in delete_message: %s", e)
            finally:
                conn.close()

        # In-memory delete
        initial_len = len(self._in_memory_messages)
        self._in_memory_messages = [
            m for m in self._in_memory_messages
            if not (
                m.get("message_id") == message_id
                and (project_id is None or m.get("project_id") == project_id)
                and (user_id is None or m.get("user_id") == user_id)
            )
        ]
        return initial_len - len(self._in_memory_messages)

    def clear_project_history(
        self,
        project_id: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """Clears all chat messages for given project_id and/or user_id."""
        if project_id is None and user_id is None:
            raise ValueError("At least project_id or user_id must be provided to clear chat history.")

        self.ensure_tables_exist()
        conn = self.get_connection()

        if conn:
            try:
                with conn:
                    with conn.cursor() as cur:
                        where_clauses = []
                        params = []
                        if project_id is not None:
                            where_clauses.append("project_id = %s")
                            params.append(project_id)
                        if user_id is not None:
                            where_clauses.append("user_id = %s::uuid")
                            params.append(user_id)

                        sql = f'DELETE FROM "Project_Chat_Message" WHERE {" AND ".join(where_clauses)};'
                        cur.execute(sql, tuple(params))
                        return cur.rowcount
            except Exception as e:
                logger.error("Error in clear_project_history: %s", e)
            finally:
                conn.close()

        # In-memory clear
        initial_len = len(self._in_memory_messages)
        self._in_memory_messages = [
            m for m in self._in_memory_messages
            if not (
                (project_id is None or m.get("project_id") == project_id)
                and (user_id is None or m.get("user_id") == user_id)
            )
        ]
        return initial_len - len(self._in_memory_messages)
