from fastapi import APIRouter, Depends, HTTPException, status
from app.api.deps import get_indexing_service
from app.modules.indexing.schemas import EmbeddingRequest, EmbeddingResponse
from app.modules.indexing.service import IndexingService

router = APIRouter(prefix="/embeddings", tags=["Phase 2: Indexing & Embeddings"])


@router.post(
    "",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate vector embeddings",
    description="Transforms text chunks into dense floating point embedding vectors.",
)
async def create_embeddings(
    request: EmbeddingRequest,
    service: IndexingService = Depends(get_indexing_service),
) -> EmbeddingResponse:
    if not request.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'texts' list cannot be empty.",
        )
    return service.generate_embeddings(request)
