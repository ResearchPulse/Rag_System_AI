from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.gateway import ChatRequest, ChatResponse
from app.services.gateway import GatewayService

router = APIRouter(prefix="/api/v1", tags=["Chat & Gateway Orchestration"])


def get_gateway_service() -> GatewayService:
    """Dependency injector for GatewayService."""
    return GatewayService()


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with RAG pipeline",
    description="Public entry point for clients. Orchestrates vector retrieval, passage reranking, context assembly, and LLM answer generation.",
    responses={
        200: {"description": "RAG chat answer generated successfully"},
        400: {"description": "Invalid query prompt"},
    },
)
async def chat_pipeline(
    request: ChatRequest,
    service: GatewayService = Depends(get_gateway_service),
) -> ChatResponse:
    """Executes end-to-end RAG workflow across microservices."""
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'query' field cannot be empty.",
        )
    return await service.chat(request)
