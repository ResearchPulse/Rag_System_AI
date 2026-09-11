"""Step 11: LLM Generation - Synthesize grounded answer conditioned on context."""
from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel


class GeneratedOutput(BaseModel):
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
        pass
