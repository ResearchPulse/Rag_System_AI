from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse
from app.services.retrieval import RetrievalService

router = APIRouter(prefix="/api/v1", tags=["Retrieval"])


def get_retrieval_service() -> RetrievalService:
    """Dependency injector for RetrievalService."""
    return RetrievalService()


@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve relevant context passages",
    description="Queries the vector database for text passages semantically related to the query and applies optional reranking.",
    responses={
        200: {"description": "Relevant context passages retrieved successfully"},
        400: {"description": "Invalid query or parameters"},
    },
)
async def retrieve_contexts(
    request: RetrievalRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> RetrievalResponse:
    """Performs semantic similarity search and reranking on knowledge base vectors."""
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'query' field cannot be empty.",
        )
    return service.retrieve_contexts(request)
