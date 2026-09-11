import math
from typing import List
from app.core.config import get_settings
from app.modules.indexing.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    EmbeddingVectorItem,
)


class IndexingService:
    """Service orchestrating Phase 2: Indexing (Step 5 Embeddings & Step 6 Vector DB)."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generates dense vector embeddings for input texts."""
        model = request.model or self.settings.EMBEDDING_MODEL
        dimension = self.settings.EMBEDDING_DIMENSION

        items: List[EmbeddingVectorItem] = []
        total_tokens = 0

        for idx, text in enumerate(request.texts):
            tokens = max(len(text.split()), 1)
            total_tokens += tokens
            seed = sum(ord(c) for c in text) % 1000
            vector = [
                round(math.sin(seed + i) / math.sqrt(dimension), 6)
                for i in range(dimension)
            ]
            items.append(EmbeddingVectorItem(index=idx, vector=vector, dimension=dimension))

        return EmbeddingResponse(
            model=model,
            dimension=dimension,
            data=items,
            usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )
