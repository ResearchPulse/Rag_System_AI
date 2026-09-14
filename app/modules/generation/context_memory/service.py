import logging
import re
from typing import Any, Dict, List, Optional

from app.modules.generation.context_memory.schemas import (
    ContextMemoryState,
    DialogueTurn,
    ResetContextMemoryResponse,
    UpdateContextMemoryRequest,
)

logger = logging.getLogger(__name__)


class ContextMemoryService:
    """Service managing sliding-window conversation memory and working context per session."""

    _sessions: Dict[str, ContextMemoryState] = {}

    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max_turns

    def _build_session_key(self, project_id: Optional[int], user_id: Optional[str]) -> str:
        p_str = str(project_id) if project_id is not None else "global"
        u_str = str(user_id) if user_id else "anonymous"
        return f"{p_str}:{u_str}"

    def get_memory(self, project_id: Optional[int], user_id: Optional[str]) -> ContextMemoryState:
        """Retrieves active conversation memory for project_id and user_id."""
        key = self._build_session_key(project_id, user_id)
        if key not in self._sessions:
            self._sessions[key] = ContextMemoryState(
                project_id=project_id,
                user_id=user_id,
            )
        return self._sessions[key]

    def record_turn(
        self,
        project_id: Optional[int],
        user_id: Optional[str],
        user_query: str,
        assistant_answer: str,
        contexts: Optional[List[Any]] = None,
    ) -> ContextMemoryState:
        """Records a new question-answer turn and updates entity/topic tracking."""
        key = self._build_session_key(project_id, user_id)
        mem = self.get_memory(project_id, user_id)

        # 1. Extract titles from contexts or answer
        articles: List[str] = []
        if contexts:
            for c in contexts:
                meta = getattr(c, "metadata", {}) or {}
                t = meta.get("title")
                if t and t not in articles:
                    articles.append(t)

        if not articles:
            # Extract bolded paper titles from assistant markdown answer
            found_titles = re.findall(r"\*\*([^\*]{10,120})\*\*", assistant_answer)
            articles.extend(found_titles[:4])

        # 2. Extract potential years
        years = [int(y) for y in re.findall(r"\b(20\d\d|19\d\d)\b", user_query)]
        active_year = years[0] if years else mem.active_year

        # 3. Extract dynamic topic
        from app.modules.retrieval.query_classifier import QueryClassifier
        extracted_topic = QueryClassifier._extract_dynamic_topic(user_query)
        active_topic = extracted_topic if extracted_topic else mem.active_topic

        # 4. Extract potential author names
        authors: List[str] = []
        author_matches = re.findall(r"(?:tác giả|author)\s+([A-Z\u00C0-\u024F\u1EA0-\u1EF9][A-Za-z\u00C0-\u024F\u1EA0-\u1EF9\s]{2,25})", user_query, re.IGNORECASE)
        authors.extend([a.strip() for a in author_matches if a.strip()])

        # Build turn
        turn = DialogueTurn(
            turn_id=mem.turn_count + 1,
            user_query=user_query,
            assistant_answer=assistant_answer[:600],  # compact snapshot
            referenced_articles=articles[:5],
            referenced_authors=authors[:5],
        )

        mem.recent_turns.append(turn)
        if len(mem.recent_turns) > self.max_turns:
            mem.recent_turns.pop(0)

        mem.turn_count += 1
        if active_topic:
            mem.active_topic = active_topic
        if active_year:
            mem.active_year = active_year
        if articles:
            mem.referenced_articles = articles[:6]
        if authors:
            mem.referenced_authors = authors[:6]

        self._sessions[key] = mem
        return mem

    def reformulate_query_with_context(
        self,
        query: str,
        project_id: Optional[int],
        user_id: Optional[str],
    ) -> str:
        """Resolves coreferences and pronouns in follow-up queries using context memory.
        
        Example:
            Turn 1: "Các bài báo về Graph RAG năm 2025"
            Turn 2: "Ai là tác giả của bài đầu tiên?"
            -> "Ai là tác giả của bài [Tên bài 1] Graph RAG 2025"
        """
        mem = self.get_memory(project_id, user_id)
        if not mem.recent_turns:
            return query

        lower_q = query.lower()

        # Indicators of conversational pronoun/coreference
        has_coref = any(
            re.search(pat, lower_q)
            for pat in [
                r"\b(đó|này|kia|trong\s*số\s*đó|những\s*bài\s*(đó|này))\b",
                r"\b(bài\s*(đó|này|ấy)|công\s*trình\s*(đó|này))\b",
                r"\b(bài\s*(thứ\s*nhất|thứ\s*1|đầu\s*tiên|thứ\s*hai|thứ\s*2|thứ\s*ba|thứ\s*3))\b",
                r"\b(tác\s*giả\s*(đó|này)|người\s*(đó|này)|họ|ông\s*ấy|bà\s*ấy)\b",
                r"\b(vừa\s*nêu|vừa\s*nói|ở\s*trên|vừa\s*đề\s*cập)\b",
            ]
        )

        if not has_coref:
            return query

        reformulated = query

        # 1. Resolving specific ordinal articles ("bài thứ nhất", "bài đầu tiên", "bài thứ hai")
        if mem.referenced_articles:
            if re.search(r"\b(bài\s*thứ\s*(nhất|1)|bài\s*đầu\s*tiên)\b", lower_q):
                target_art = mem.referenced_articles[0]
                reformulated = f"{reformulated} (bài báo: \"{target_art}\")"
            elif re.search(r"\b(bài\s*thứ\s*(hai|2))\b", lower_q) and len(mem.referenced_articles) > 1:
                target_art = mem.referenced_articles[1]
                reformulated = f"{reformulated} (bài báo: \"{target_art}\")"
            elif re.search(r"\b(bài\s*thứ\s*(ba|3))\b", lower_q) and len(mem.referenced_articles) > 2:
                target_art = mem.referenced_articles[2]
                reformulated = f"{reformulated} (bài báo: \"{target_art}\")"
            elif re.search(r"\b(bài\s*(đó|này|ấy)|công\s*trình\s*(đó|này))\b", lower_q):
                target_art = mem.referenced_articles[0]
                reformulated = f"{reformulated} (bài báo: \"{target_art}\")"

        # 2. Resolving general topic context ("trong số đó", "về chủ đề đó")
        if mem.active_topic and mem.active_topic.lower() not in lower_q:
            reformulated = f"{reformulated} (chủ đề: {mem.active_topic})"

        # 3. Resolving year context
        if mem.active_year and not re.search(r"\b(19\d\d|20\d\d)\b", query):
            reformulated = f"{reformulated} (năm: {mem.active_year})"

        logger.info("[Context Memory] Query reformulated: '%s' -> '%s'", query, reformulated)
        return reformulated

    def format_history_for_prompt(
        self,
        project_id: Optional[int],
        user_id: Optional[str],
        max_turns: int = 3,
    ) -> str:
        """Formats recent dialogue turns into a prompt block for multi-turn LLM generation."""
        mem = self.get_memory(project_id, user_id)
        if not mem.recent_turns:
            return ""

        turns_to_include = mem.recent_turns[-max_turns:]
        lines = ["--- LỊCH SỬ TRAO ĐỔI TRƯỚC ĐÓ TRONG PHIÊN (CONVERSATION CONTEXT) ---"]
        for t in turns_to_include:
            lines.append(f"User: {t.user_query}")
            ans_snippet = t.assistant_answer.strip().replace("\n", " ")
            if len(ans_snippet) > 200:
                ans_snippet = ans_snippet[:200] + "..."
            lines.append(f"Assistant: {ans_snippet}")
        lines.append("--------------------------------------------------------------------")
        return "\n".join(lines)

    def update_memory(self, req: UpdateContextMemoryRequest) -> ContextMemoryState:
        """Manually updates or sets focus entities for a user session."""
        mem = self.get_memory(req.project_id, req.user_id)
        if req.active_topic is not None:
            mem.active_topic = req.active_topic
        if req.active_year is not None:
            mem.active_year = req.active_year
        if req.referenced_articles is not None:
            mem.referenced_articles = req.referenced_articles
        if req.referenced_authors is not None:
            mem.referenced_authors = req.referenced_authors

        key = self._build_session_key(req.project_id, req.user_id)
        self._sessions[key] = mem
        return mem

    def reset_memory(self, project_id: Optional[int], user_id: Optional[str]) -> ResetContextMemoryResponse:
        """Clears conversational memory state for this user/project session."""
        key = self._build_session_key(project_id, user_id)
        if key in self._sessions:
            del self._sessions[key]
        return ResetContextMemoryResponse(
            success=True,
            message="Đã xóa và làm mới bộ nhớ ngữ cảnh trò chuyện thành công.",
        )
