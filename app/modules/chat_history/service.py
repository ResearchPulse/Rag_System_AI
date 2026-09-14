import logging
import uuid
from typing import Optional

from app.modules.chat_history.repository import ChatHistoryRepository
from app.modules.chat_history.schemas import (
    ChatMessageCreate,
    ChatMessageItem,
    ChatMessageRole,
    ChatMessageStatus,
    ChatHistoryResponse,
    DeleteHistoryResponse,
)

logger = logging.getLogger(__name__)


class ChatHistoryService:
    """Service layer managing Chat Message history operations and business logic."""

    def __init__(self, repository: Optional[ChatHistoryRepository] = None) -> None:
        self.repo = repository or ChatHistoryRepository()

    def _normalize_uuid(self, user_id: str) -> str:
        """Validates or standardizes UUID string format."""
        try:
            return str(uuid.UUID(user_id))
        except ValueError:
            # Fallback to generating consistent uuid5 if non-uuid string passed
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, user_id))

    def save_message(self, data: ChatMessageCreate) -> ChatMessageItem:
        """Saves a single user or assistant chat message."""
        normalized_user_id = self._normalize_uuid(data.user_id)
        data.user_id = normalized_user_id
        return self.repo.create_message(data)

    def record_chat_turn(
        self,
        project_id: Optional[int],
        user_id: str,
        user_query: str,
        assistant_answer: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        latency_ms: Optional[int] = None,
    ) -> tuple[int, int]:
        """Convenience method to save both USER and ASSISTANT messages in one turn.
        
        Returns (user_message_id, assistant_message_id).
        """
        norm_user_id = self._normalize_uuid(user_id)

        # 1. Save user message
        user_msg = self.repo.create_message(
            ChatMessageCreate(
                project_id=project_id,
                user_id=norm_user_id,
                role=ChatMessageRole.USER,
                content=user_query,
                status=ChatMessageStatus.COMPLETED,
            )
        )

        # 2. Save assistant response
        asst_msg = self.repo.create_message(
            ChatMessageCreate(
                project_id=project_id,
                user_id=norm_user_id,
                role=ChatMessageRole.ASSISTANT,
                content=assistant_answer,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                status=ChatMessageStatus.COMPLETED,
            )
        )

        return user_msg.message_id, asst_msg.message_id

    def get_history(
        self,
        project_id: Optional[int] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        order: str = "asc",
    ) -> ChatHistoryResponse:
        """Fetches paginated chat messages for given project / user."""
        norm_user_id = self._normalize_uuid(user_id) if user_id else None
        limit = max(1, min(limit, 100))
        offset = max(0, offset)

        messages, total = self.repo.get_messages(
            project_id=project_id,
            user_id=norm_user_id,
            limit=limit,
            offset=offset,
            order=order,
        )
        return ChatHistoryResponse(
            total=total,
            limit=limit,
            offset=offset,
            messages=messages,
        )

    def get_message(
        self,
        message_id: int,
        project_id: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> Optional[ChatMessageItem]:
        """Retrieves a single message by ID."""
        norm_user_id = self._normalize_uuid(user_id) if user_id else None
        return self.repo.get_message_by_id(message_id, project_id, norm_user_id)

    def delete_message(
        self,
        message_id: int,
        project_id: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> DeleteHistoryResponse:
        """Deletes a specific message."""
        norm_user_id = self._normalize_uuid(user_id) if user_id else None
        count = self.repo.delete_message(message_id, project_id, norm_user_id)
        return DeleteHistoryResponse(
            deleted_count=count,
            message=f"Đã xóa thành công {count} tin nhắn." if count > 0 else "Không tìm thấy tin nhắn cần xóa.",
        )

    def clear_history(
        self,
        project_id: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> DeleteHistoryResponse:
        """Clears all conversation messages for project / user."""
        norm_user_id = self._normalize_uuid(user_id) if user_id else None
        count = self.repo.clear_project_history(project_id, norm_user_id)
        return DeleteHistoryResponse(
            deleted_count=count,
            message=f"Đã xóa toàn bộ {count} tin nhắn trong lịch sử trò chuyện.",
        )
