"""Step 7: Query Rewriting - Expand intent, clarify, and rewrite for better match."""
from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel


class RewrittenQuery(BaseModel):
    original_query: str
    rewritten_query: str
    expanded_queries: List[str] = []
    extracted_keywords: List[str] = []


class BaseQueryRewriter(ABC):
    """Abstract interface for Step 7: Query Rewriting."""
    @abstractmethod
    async def rewrite(self, query: str) -> RewrittenQuery:
        pass
