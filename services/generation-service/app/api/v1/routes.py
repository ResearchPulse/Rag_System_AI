from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.generation import GenerationRequest, GenerationResponse
from app.services.generation import GenerationService

router = APIRouter(prefix="/api/v1", tags=["Generation"])


def get_generation_service() -> GenerationService:
    """Dependency injector for GenerationService."""
    return GenerationService()


@router.post(
    "/generate",
    response_model=GenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate answer from context",
    description="Synthesizes a response to a user query grounded strictly in provided context passages using an LLM.",
    responses={
        200: {"description": "Grounded answer synthesized successfully"},
        400: {"description": "Missing query or context chunks"},
    },
)
async def generate_response(
    request: GenerationRequest,
    service: GenerationService = Depends(get_generation_service),
) -> GenerationResponse:
    """Invokes LLM synthesis pipeline with contextual grounding."""
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'query' field cannot be empty.",
        )
    if not request.contexts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one context passage must be provided in 'contexts'.",
        )
    return service.generate_answer(request)
