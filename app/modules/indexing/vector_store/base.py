"""Step 6: Vector Database - Store embeddings in vector DB for fast intelligent retrieval."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class VectorRecord(BaseModel):
    id: str
    vector: List[float]
    payload: Dict[str, Any] = {}


class VectorSearchResult(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any] = {}


class BaseVectorStore(ABC):
    """Abstract interface for Step 6: Vector Database (Qdrant, PgVector, Milvus, Chroma)."""
    @abstractmethod
    async def upsert(self, collection_name: str, records: List[VectorRecord]) -> bool:
        pass

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        pass
