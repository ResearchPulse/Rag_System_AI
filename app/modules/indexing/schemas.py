from typing import List, Optional
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request schema for generating vector embeddings."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        description="List of text chunks to convert into dense vector embeddings",
        examples=[["Scholarly trend tracking in AI", "Modular Monolith RAG architecture"]],
    )
    model: Optional[str] = Field(
        default=None,
        description="Model override",
        examples=["text-embedding-3-small"],
    )


class EmbeddingVectorItem(BaseModel):
    """Individual vector output item."""
    index: int = Field(..., examples=[0])
    vector: List[float] = Field(...)
    dimension: int = Field(..., examples=[1536])


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = Field(..., examples=[18])
    total_tokens: int = Field(..., examples=[18])


class EmbeddingResponse(BaseModel):
    model: str = Field(..., examples=["text-embedding-3-small"])
    dimension: int = Field(..., examples=[1536])
    data: List[EmbeddingVectorItem] = Field(default_factory=list)
    usage: EmbeddingUsage
