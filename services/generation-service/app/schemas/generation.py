from typing import List, Optional
from pydantic import BaseModel, Field


class ContextItem(BaseModel):
    """Schema representing an individual context fragment passed to the LLM."""
    id: Optional[str] = Field(default=None, description="Identifier of the context chunk", examples=["chk_001"])
    content: str = Field(..., description="Context passage content", examples=["RAG systems combine neural retrieval with generative language models."])
    source: Optional[str] = Field(default=None, description="Document source or reference title", examples=["State of Scientific AI Publications 2026"])


class GenerationRequest(BaseModel):
    """Request schema for generating an answer from retrieved context."""
    query: str = Field(
        ...,
        min_length=1,
        description="The original user query or research question",
        examples=["What are the recent trends in transformer models for scientific literature?"],
    )
    contexts: List[ContextItem] = Field(
        ...,
        min_length=1,
        description="List of context chunks to ground the model response",
    )
    model: Optional[str] = Field(
        default=None,
        description="Target LLM model (e.g., gpt-4o-mini, llama3, mistral)",
        examples=["gpt-4o-mini"],
    )
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature", examples=[0.7])
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096, description="Maximum completion tokens", examples=[1024])
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system instruction to guide answer style",
        examples=["You are an expert scientific literature assistant. Ground your answer strictly on provided contexts."],
    )


class GenerationUsage(BaseModel):
    """Token usage metrics for LLM generation."""
    prompt_tokens: int = Field(..., description="Tokens consumed by prompt and context", examples=[185])
    completion_tokens: int = Field(..., description="Tokens generated in response", examples=[88])
    total_tokens: int = Field(..., description="Total tokens utilized", examples=[273])


class GenerationResponse(BaseModel):
    """Response schema containing the synthesized answer and metadata."""
    query: str = Field(..., description="Original query", examples=["What are the recent trends in transformer models for scientific literature?"])
    answer: str = Field(..., description="Synthesized response from the LLM", examples=["Based on recent publications, transformer models focus on RAG optimization..."])
    model: str = Field(..., description="Model used for generation", examples=["gpt-4o-mini"])
    contexts_used: int = Field(..., description="Number of context chunks incorporated", examples=[2])
    citations: List[str] = Field(default_factory=list, description="Extracted citations or source references")
    usage: GenerationUsage = Field(..., description="Detailed token consumption statistics")
    latency_ms: float = Field(..., description="Generation latency in milliseconds", examples=[320.4])

