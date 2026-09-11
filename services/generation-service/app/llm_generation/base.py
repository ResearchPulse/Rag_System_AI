"""Step 11: LLM Generation - LLM synthesizes grounded answer using provided context."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class GeneratedOutput(BaseModel):
    """Represents the raw generated LLM answer with token metrics."""
    answer: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    raw_response: Optional[Any] = None


class BaseLLMGenerator(ABC):
    """Abstract interface for Step 11: LLM Generation."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> GeneratedOutput:
        """Invokes generative model to synthesize grounded response."""
        pass
