from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessageRole(str, Enum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"


class ChatMessageStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class ChatMessageCreate(BaseModel):
    """Payload to create a new chat message record."""
    project_id: Optional[int] = Field(default=None, description="Associated project ID", examples=[12])
    user_id: str = Field(..., description="UUID of the user", examples=["550e8400-e29b-41d4-a716-446655440000"])
    role: ChatMessageRole = Field(default=ChatMessageRole.USER, description="Message sender role")
    content: str = Field(..., min_length=1, description="Message text content")
    model: Optional[str] = Field(default=None, description="LLM model identifier", examples=["llama3.2:3b"])
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: Optional[int] = Field(default=None, ge=0)
    status: ChatMessageStatus = Field(default=ChatMessageStatus.COMPLETED)


class ChatMessageItem(BaseModel):
    """Output representation of a stored chat message."""
    message_id: int
    project_id: Optional[int] = None
    user_id: str
    role: str
    content: str
    model: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: Optional[int] = None
    status: str
    created_at: Optional[datetime] = None


class ChatHistoryResponse(BaseModel):
    """Paginated response containing chat history messages."""
    total: int
    limit: int
    offset: int
    messages: List[ChatMessageItem] = Field(default_factory=list)


class DeleteHistoryResponse(BaseModel):
    """Response returned upon deleting chat history."""
    deleted_count: int
    message: str
