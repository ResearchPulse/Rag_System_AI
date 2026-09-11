"""Step 9: Reranking - Rerank results using a cross-encoder model for high-precision relevance."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel


class RerankedCandidate(BaseModel):
    """Represents a passage candidate with cross-encoder relevance score."""
    chunk_id: str
    content: str
    original_score: float
    rerank_score: float
    metadata: Dict[str, Any] = {}


class BaseReranker(ABC):
    """Abstract interface for Step 9: Reranking (e.g. Cohere Rerank, BGE-Reranker, Cross-Encoder)."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RerankedCandidate],
        top_n: int = 5,
    ) -> List[RerankedCandidate]:
        """Reranks candidate chunks based on true query-document semantic relevance."""
        pass
