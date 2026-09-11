from fastapi import APIRouter, Depends, HTTPException, Path, status
from app.api.deps import get_ingestion_service
from app.modules.ingestion.schemas import (
    DocumentDetailResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
)
from app.modules.ingestion.service import IngestionService

router = APIRouter(prefix="/documents", tags=["Phase 1: Ingestion"])


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and chunk scholarly document",
    description="Accepts document text, splits it into semantic chunks, and extracts initial metadata.",
)
async def upload_document(
    request: DocumentUploadRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentUploadResponse:
    if not request.title:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document title is required.",
        )
    return service.process_document(request)


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details and chunks",
    description="Retrieves document metadata, processing status, and text chunks.",
)
async def get_document(
    document_id: str = Path(..., description="Unique document ID", examples=["doc_998877"]),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentDetailResponse:
    return service.get_document(document_id)
