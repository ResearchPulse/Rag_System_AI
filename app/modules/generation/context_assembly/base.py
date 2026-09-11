"""Step 10: Context Assembly - Select, order, and compress the best context for the prompt."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel


class AssembledContext(BaseModel):
    formatted_prompt_context: str
    selected_chunk_ids: List[str]
    total_tokens: int
    citations_metadata: List[Dict[str, Any]] = []


class BaseContextAssembler(ABC):
    """Abstract interface for Step 10: Context Assembly."""
    @abstractmethod
    def assemble(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_context_tokens: int = 2048,
    ) -> AssembledContext:
        pass
