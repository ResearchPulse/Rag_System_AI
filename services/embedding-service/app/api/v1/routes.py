from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.embedding import EmbeddingRequest, EmbeddingResponse
from app.services.embedding import EmbeddingService

router = APIRouter(prefix="/api/v1", tags=["Embeddings"])


def get_embedding_service() -> EmbeddingService:
    """Dependency injector for EmbeddingService."""
    return EmbeddingService()


@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate vector embeddings",
    description="Transforms an array of text chunks or queries into dense float embedding vectors.",
    responses={
        200: {"description": "Embeddings successfully generated"},
        400: {"description": "Empty or invalid input texts"},
    },
)
async def create_embeddings(
    request: EmbeddingRequest,
    service: EmbeddingService = Depends(get_embedding_service),
) -> EmbeddingResponse:
    """Creates dense vector representations for input strings."""
    if not request.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'texts' list cannot be empty.",
        )
    return service.generate_embeddings(request)
