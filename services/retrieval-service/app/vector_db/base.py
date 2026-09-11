"""Step 6: Vector Database - Store embeddings in a vector database for fast, intelligent retrieval."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class VectorRecord(BaseModel):
    """Represents a vector record to store or query in the vector database."""
    id: str
    vector: List[float]
    payload: Dict[str, Any] = {}


class VectorSearchResult(BaseModel):
    """Represents a similarity search result from the vector database."""
    id: str
    score: float
    payload: Dict[str, Any] = {}


class BaseVectorDatabase(ABC):
    """Abstract interface for Step 6: Vector Database (e.g. Qdrant, Milvus, Chroma, PgVector)."""

    @abstractmethod
    async def upsert(self, collection_name: str, records: List[VectorRecord]) -> bool:
        """Stores or updates vector embeddings with payloads."""
        pass

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Performs nearest-neighbor vector similarity search."""
        pass
