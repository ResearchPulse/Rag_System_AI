"""Step 4: Metadata Extraction - Extract title, author, year, quartile, and relations."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel


class ExtractedMetadata(BaseModel):
    title: str = ""
    authors: List[str] = []
    year: int = 0
    venue: str = ""
    quartile: str = ""
    keywords: List[str] = []
    attributes: Dict[str, Any] = {}


class BaseMetadataExtractor(ABC):
    """Abstract interface for Step 4: Metadata Extraction."""
    @abstractmethod
    def extract(self, text: str, initial_metadata: Dict[str, Any] = None) -> ExtractedMetadata:
        pass
