from fastapi import APIRouter, Depends, HTTPException, status
from app.api.deps import get_generation_service
from app.modules.generation.schemas import GenerationRequest, GenerationResponse
from app.modules.generation.service import GenerationService

router = APIRouter(prefix="/generate", tags=["Phase 4: Generation & Evaluation"])


@router.post(
    "",
    response_model=GenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate grounded answer from context",
    description="Synthesizes a citation-grounded response using an LLM conditioned on assembled context.",
)
async def generate_response(
    request: GenerationRequest,
    service: GenerationService = Depends(get_generation_service),
) -> GenerationResponse:
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'query' field cannot be empty.",
        )
    if not request.contexts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one context passage is required.",
        )
    return service.generate(request)
