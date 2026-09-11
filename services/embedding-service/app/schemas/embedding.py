from typing import List, Optional
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request schema for generating vector embeddings."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        description="List of text strings or chunk passages to be converted into vector embeddings",
        examples=[["Scientific publication trends in machine learning", "RAG architecture with FastAPI microservices"]],
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional embedding model override (e.g. text-embedding-3-small, BAAI/bge-m3)",
        examples=["text-embedding-3-small"],
    )


class EmbeddingVectorItem(BaseModel):
    """Schema representing an individual vector output."""
    index: int = Field(..., description="Position index of the corresponding input text", examples=[0])
    vector: List[float] = Field(..., description="High-dimensional floating point embedding vector")
    dimension: int = Field(..., description="Dimensionality of the vector", examples=[1536])


class EmbeddingUsage(BaseModel):
    """Token usage metrics for embedding generation."""
    prompt_tokens: int = Field(..., description="Estimated token count of the input", examples=[24])
    total_tokens: int = Field(..., description="Total tokens processed", examples=[24])


class EmbeddingResponse(BaseModel):
    """Response schema returned after embedding generation."""
    model: str = Field(..., description="Name of the model used to compute embeddings", examples=["text-embedding-3-small"])
    dimension: int = Field(..., description="Vector dimensionality", examples=[1536])
    data: List[EmbeddingVectorItem] = Field(..., description="Array of embedding vectors")
    usage: EmbeddingUsage = Field(..., description="Token usage details")

