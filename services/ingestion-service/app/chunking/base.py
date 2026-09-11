"""Step 3: Meaningful Chunking - Split documents into semantic, coherent chunks."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel


class TextChunk(BaseModel):
    """Represents a semantically meaningful text chunk."""
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = {}


class BaseChunker(ABC):
    """Abstract interface for Step 3: Meaningful Chunking."""

    @abstractmethod
    def chunk(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """Splits text content into semantic chunks preserving context."""
        pass
