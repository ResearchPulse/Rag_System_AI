from fastapi import APIRouter, Depends, HTTPException, status
from app.api.deps import get_generation_service, get_retrieval_service
from app.modules.generation.schemas import RagPipelineRequest, RagPipelineResponse
from app.modules.generation.service import GenerationService
from app.modules.retrieval.service import RetrievalService

router = APIRouter(prefix="/chat", tags=["End-to-End RAG Pipeline"])


@router.post(
    "",
    response_model=RagPipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with RAG pipeline",
    description="Unified entry point. Coordinates retrieval, hybrid search, reranking, context assembly, and LLM generation in-process.",
)
async def chat_pipeline(
    request: RagPipelineRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service),
) -> RagPipelineResponse:
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'query' field cannot be empty.",
        )
    return generation_service.execute_rag_pipeline(request, retrieval_service)
