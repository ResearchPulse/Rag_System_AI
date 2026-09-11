"""Step 5: Embeddings - Convert chunks into vector embeddings capturing deep semantics."""
from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel


class VectorOutput(BaseModel):
    index: int
    vector: List[float]
    dimension: int


class BaseEmbedder(ABC):
    """Abstract interface for Step 5: Embeddings."""
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[VectorOutput]:
        pass
