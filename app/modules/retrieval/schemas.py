from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalFilter(BaseModel):
    document_ids: Optional[List[str]] = Field(default=None, examples=[["doc_998877"]])
    categories: Optional[List[str]] = Field(default=None, examples=[["AI", "Scientometrics"]])
    metadata: Optional[Dict[str, Any]] = None


class RetrievalRequest(BaseModel):
    """Request schema for top-k contextual retrieval."""
    query: str = Field(
        ...,
        min_length=1,
        description="Search query to retrieve context for",
        examples=["What are recent publication trends in generative AI models?"],
    )
    top_k: int = Field(default=5, ge=1, le=50, examples=[5])
    score_threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, examples=[0.5])
    rerank: bool = Field(default=True, examples=[True])
    filter: Optional[RetrievalFilter] = None


class RetrievedChunk(BaseModel):
    chunk_id: str = Field(..., examples=["chk_102030"])
    document_id: str = Field(..., examples=["doc_998877"])
    content: str = Field(..., examples=["Generative AI papers grew significantly across scholarly domains."])
    score: float = Field(..., examples=[0.91])
    rerank_score: Optional[float] = Field(default=None, examples=[0.95])
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    query: str = Field(..., examples=["What are recent publication trends in generative AI models?"])
    total_found: int = Field(..., examples=[2])
    results: List[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float = Field(..., examples=[12.4])
