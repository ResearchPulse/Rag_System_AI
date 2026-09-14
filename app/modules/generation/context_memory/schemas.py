from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class DialogueTurn(BaseModel):
    """Represents a single conversational turn between user and assistant."""
    turn_id: int
    user_query: str
    assistant_answer: str
    referenced_articles: List[str] = Field(default_factory=list)
    referenced_authors: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class ContextMemoryState(BaseModel):
    """Represents the active working memory and entities for a user session."""
    project_id: Optional[int] = None
    user_id: Optional[str] = None
    active_topic: Optional[str] = None
    active_year: Optional[int] = None
    referenced_articles: List[str] = Field(default_factory=list)
    referenced_authors: List[str] = Field(default_factory=list)
    recent_turns: List[DialogueTurn] = Field(default_factory=list)
    turn_count: int = 0


class UpdateContextMemoryRequest(BaseModel):
    """Payload to manually update the active context memory state."""
    project_id: Optional[int] = Field(default=None, description="Optional project ID", examples=[12])
    user_id: str = Field(..., description="UUID of the user", examples=["550e8400-e29b-41d4-a716-446655440000"])
    active_topic: Optional[str] = Field(default=None, description="Set focus research topic", examples=["Graph RAG"])
    active_year: Optional[int] = Field(default=None, description="Set focus publication year", examples=[2025])
    referenced_articles: Optional[List[str]] = Field(default=None, description="List of focus article titles")
    referenced_authors: Optional[List[str]] = Field(default=None, description="List of focus authors")


class ResetContextMemoryResponse(BaseModel):
    """Response returned upon clearing context memory."""
    success: bool
    message: str
