"""Step 9: Reranking - Cross-encoder precision scoring to avoid volume over relevance."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel


class RerankedCandidate(BaseModel):
    chunk_id: str
    content: str
    original_score: float
    rerank_score: float
    metadata: Dict[str, Any] = {}


class BaseReranker(ABC):
    """Abstract interface for Step 9: Reranking."""
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RerankedCandidate],
        top_n: int = 5,
    ) -> List[RerankedCandidate]:
        pass
