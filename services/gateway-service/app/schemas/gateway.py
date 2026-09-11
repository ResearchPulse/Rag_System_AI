from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Client request schema for chatting with the RAG pipeline."""
    query: str = Field(
        ...,
        min_length=1,
        description="User question or search prompt",
        examples=["What are the latest publication trends in AI scientific literature?"],
    )
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of context passages to retrieve", examples=[5])
    model: Optional[str] = Field(default=None, description="Optional LLM model override", examples=["gpt-4o-mini"])
    include_contexts: bool = Field(default=True, description="Whether to include retrieved context excerpts in response", examples=[True])
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="LLM sampling temperature", examples=[0.7])


class ChatContextSource(BaseModel):
    """Schema representing an excerpt of context retrieved and used in the answer."""
    chunk_id: str = Field(..., description="Chunk identifier", examples=["chk_trend_001"])
    document_id: str = Field(..., description="Parent document identifier", examples=["doc_scientometrics_2026"])
    content: str = Field(..., description="Excerpt content from chunk", examples=["Publication trends analysis..."])
    score: float = Field(..., description="Relevance similarity score", examples=[0.92])
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the source chunk")


class ChatExecutionTiming(BaseModel):
    """Latency metrics for individual pipeline steps."""
    retrieval_ms: float = Field(..., description="Duration of vector search and reranking in ms", examples=[14.5])
    generation_ms: float = Field(..., description="Duration of LLM response generation in ms", examples=[320.4])
    total_ms: float = Field(..., description="Total end-to-end processing time in ms", examples=[334.9])


class ChatResponse(BaseModel):
    """Final unified response returned to the client by the API Gateway."""
    query: str = Field(..., description="Original user prompt", examples=["What are the latest publication trends in AI scientific literature?"])
    answer: str = Field(..., description="Final synthesized and grounded answer", examples=["Based on recent scientific publications..."])
    model: str = Field(..., description="Model leveraged to generate the answer", examples=["gpt-4o-mini"])
    contexts: Optional[List[ChatContextSource]] = Field(default=None, description="List of source context chunks if requested")
    citations: List[str] = Field(default_factory=list, description="Unique source citations referenced in the answer")
    timing: ChatExecutionTiming = Field(..., description="Execution timing breakdown")

