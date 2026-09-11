from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalFilter(BaseModel):
    """Optional metadata filters for vector retrieval."""
    document_ids: Optional[List[str]] = Field(default=None, description="Restrict search to specific document IDs", examples=[["doc_123456789"]])
    categories: Optional[List[str]] = Field(default=None, description="Filter by document subject domain/category", examples=[["AI", "Computer Science"]])
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Custom metadata filter key-values")


class RetrievalRequest(BaseModel):
    """Request schema for querying top-k relevant contexts."""
    query: str = Field(
        ...,
        min_length=1,
        description="User search query or question to retrieve context for",
        examples=["What are the recent trends in transformer models for scientific literature?"],
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of top matching context chunks to return", examples=[5])
    score_threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score threshold", examples=[0.5])
    rerank: bool = Field(default=True, description="Whether to apply cross-encoder reranking to initial results", examples=[True])
    filter: Optional[RetrievalFilter] = Field(default=None, description="Optional metadata filter parameters")


class RetrievedChunk(BaseModel):
    """Schema representing a retrieved context passage with similarity score."""
    chunk_id: str = Field(..., description="Unique chunk ID", examples=["chk_987654321"])
    document_id: str = Field(..., description="Source document ID", examples=["doc_123456789"])
    content: str = Field(..., description="Retrieved passage text", examples=["Recent studies highlight exponential growth in generative AI publications."])
    score: float = Field(..., description="Vector similarity score (cosine/dot product)", examples=[0.895])
    rerank_score: Optional[float] = Field(default=None, description="Refined cross-encoder reranking score", examples=[0.942])
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Associated chunk and document metadata")


class RetrievalResponse(BaseModel):
    """Response schema containing ranked context passages."""
    query: str = Field(..., description="Original input query", examples=["What are the recent trends in transformer models for scientific literature?"])
    total_found: int = Field(..., description="Number of results matching the threshold", examples=[2])
    results: List[RetrievedChunk] = Field(default_factory=list, description="Ranked list of context chunks")
    latency_ms: float = Field(..., description="Retrieval and reranking latency in milliseconds", examples=[14.5])

