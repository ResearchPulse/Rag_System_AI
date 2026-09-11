"""Step 5: Embeddings - Convert chunks into vector embeddings that capture meaning."""
from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel


class VectorOutput(BaseModel):
    """Vector embedding output for a text input."""
    index: int
    vector: List[float]
    dimension: int


class BaseEmbedder(ABC):
    """Abstract interface for Step 5: Embeddings."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[VectorOutput]:
        """Converts a batch of texts into high-dimensional vectors."""
        pass
