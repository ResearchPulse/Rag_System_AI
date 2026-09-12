from fastapi import APIRouter, Depends, HTTPException, status
from app.api.deps import get_retrieval_service
from app.modules.retrieval.schemas import RetrievalRequest, RetrievalResponse
from app.modules.retrieval.service import RetrievalService
from app.modules.retrieval.query_classifier import ClassificationResult

router = APIRouter(prefix="/retrieve", tags=["Phase 3: Retrieval & Reranking"])


@router.post(
    "",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve relevant context passages",
    description="Performs semantic vector lookup, hybrid search, and cross-encoder reranking.",
)
async def retrieve_contexts(
    request: RetrievalRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> RetrievalResponse:
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'query' field cannot be empty.",
        )
    return service.retrieve(request)


@router.post(
    "/classify",
    response_model=ClassificationResult,
    status_code=status.HTTP_200_OK,
    summary="Classify user query (Query Classification Layer)",
    description="Analyzes the research query into direct_lookup, relational_reasoning, hybrid, or chitchat with routing flags and filters.",
)
async def classify_query(
    request: RetrievalRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> ClassificationResult:
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'query' field cannot be empty.",
        )
    return service.classifier.classify(request.query)

