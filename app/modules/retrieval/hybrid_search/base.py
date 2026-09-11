"""Step 8: Hybrid Search - Combine semantic (dense vector) and keyword (lexical BM25) search."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class HybridSearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    combined_score: float = 0.0
    metadata: Dict[str, Any] = {}


class BaseHybridSearcher(ABC):
    """Abstract interface for Step 8: Hybrid Search."""
    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[HybridSearchResult]:
        pass
