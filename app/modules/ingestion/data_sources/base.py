"""Step 1: Data Sources - Collect raw documents from PDFs, APIs, websites, and transcripts."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class RawDataSource(BaseModel):
    source_type: str  # "pdf", "api", "web", "transcript"
    uri: Optional[str] = None
    raw_content: Any = None
    metadata: Dict[str, Any] = {}


class BaseDataSource(ABC):
    """Abstract interface for Step 1: Data Sources."""
    @abstractmethod
    async def fetch(self, **kwargs) -> List[RawDataSource]:
        pass
