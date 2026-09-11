"""Step 4: Metadata Extraction - Extract title, author, tags, relations, and publication metadata."""
from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel


class ExtractedMetadata(BaseModel):
    """Enriched metadata extracted from scholarly content."""
    title: str = ""
    authors: list[str] = []
    year: int = 0
    venue: str = ""
    domain: str = ""
    keywords: list[str] = []
    attributes: Dict[str, Any] = {}


class BaseMetadataExtractor(ABC):
    """Abstract interface for Step 4: Metadata Extraction."""

    @abstractmethod
    def extract(self, text: str, initial_metadata: Dict[str, Any] = None) -> ExtractedMetadata:
        """Extracts structured metadata and relations from document text."""
        pass
