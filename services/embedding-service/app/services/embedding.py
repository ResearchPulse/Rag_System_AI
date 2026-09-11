import math
from typing import List
from app.core.config import get_settings
from app.schemas.embedding import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    EmbeddingVectorItem,
)

settings = get_settings()


class EmbeddingService:
    """Service handling text embedding generation."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generates embedding vectors for the requested texts."""
        # TODO: Replace with real embedding inference (OpenAI text-embedding-3, HuggingFace, Ollama, etc.)
        model = request.model or self.settings.DEFAULT_MODEL
        dimension = self.settings.EMBEDDING_DIMENSION

        items: List[EmbeddingVectorItem] = []
        total_tokens = 0

        for idx, text in enumerate(request.texts):
            # Mock deterministic float vector
            tokens = max(len(text.split()), 1)
            total_tokens += tokens
            
            # Generate deterministic mock values based on char ordinals
            seed = sum(ord(c) for c in text) % 1000
            vector = [
                round(math.sin(seed + i) / math.sqrt(dimension), 6)
                for i in range(dimension)
            ]
            
            items.append(
                EmbeddingVectorItem(
                    index=idx,
                    vector=vector,
                    dimension=dimension,
                )
            )

        return EmbeddingResponse(
            model=model,
            dimension=dimension,
            data=items,
            usage=EmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            ),
        )
