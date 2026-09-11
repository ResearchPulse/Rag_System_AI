"""Step 2: Document Loading - Ingest and parse content from diverse sources."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel


class LoadedDocument(BaseModel):
    doc_id: str
    text: str
    metadata: Dict[str, Any] = {}


class BaseDocumentLoader(ABC):
    """Abstract interface for Step 2: Document Loading."""
    @abstractmethod
    def load(self, source_path_or_stream: Any) -> List[LoadedDocument]:
        pass
