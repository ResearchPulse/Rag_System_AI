from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ContextItem(BaseModel):
    id: Optional[str] = Field(default=None, examples=["chk_pub_01"])
    content: str = Field(..., examples=["RAG combines neural retrieval with generative language models."])
    source: Optional[str] = Field(default=None, examples=["State of Scientific AI Publications 2026"])
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GenerationRequest(BaseModel):
    """Request schema for generating grounded LLM responses."""
    query: str = Field(
        ...,
        min_length=1,
        description="The research question to answer",
        examples=["What are recent publication trends in generative AI models?"],
    )
    contexts: List[ContextItem] = Field(
        default_factory=list,
        description="Context passages to ground the response",
    )
    model: Optional[str] = Field(default=None, examples=["gpt-4o-mini"])
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, examples=[0.7])
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096, examples=[1024])
    system_prompt: Optional[str] = None
    history_context: Optional[str] = None



class GenerationUsage(BaseModel):
    prompt_tokens: int = Field(..., examples=[165])
    completion_tokens: int = Field(..., examples=[82])
    total_tokens: int = Field(..., examples=[247])


class GenerationResponse(BaseModel):
    query: str = Field(..., examples=["What are recent publication trends in generative AI models?"])
    answer: str = Field(...)
    model: str = Field(..., examples=["gpt-4o-mini"])
    contexts_used: int = Field(..., examples=[2])
    citations: List[str] = Field(default_factory=list)
    usage: GenerationUsage
    latency_ms: float = Field(..., examples=[280.5])


class RagPipelineRequest(BaseModel):
    """End-to-End Chat / RAG query request for the entire pipeline."""
    query: str = Field(
        ...,
        min_length=1,
        description="User research prompt",
        examples=["Xu hướng công bố bài báo khoa học về RAG trong năm 2025-2026 là gì?"],
    )
    top_k: Optional[int] = Field(default=5, ge=1, le=20, examples=[5])
    model: Optional[str] = Field(default=None, examples=["gpt-4o-mini"])
    include_contexts: bool = Field(default=True, examples=[True])
    temperature: Optional[float] = Field(default=0.7, examples=[0.7])
    project_id: Optional[int] = Field(default=None, description="Associated project ID", examples=[12])
    user_id: Optional[str] = Field(default=None, description="UUID of the user", examples=["550e8400-e29b-41d4-a716-446655440000"])
    save_history: bool = Field(default=True, description="Whether to persist conversation history", examples=[True])


class RagPipelineResponse(BaseModel):
    """End-to-End unified response with grounded answer, citations, and metrics."""
    query: str
    answer: str
    model: str
    contexts: Optional[List[Dict[str, Any]]] = None
    citations: List[str] = Field(default_factory=list)
    latency_breakdown: Dict[str, float] = Field(default_factory=dict)
    user_message_id: Optional[int] = Field(default=None, description="ID of saved user message", examples=[101])
    assistant_message_id: Optional[int] = Field(default=None, description="ID of saved assistant message", examples=[102])

