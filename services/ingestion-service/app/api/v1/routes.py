from fastapi import APIRouter, Depends, HTTPException, Path, status
from app.schemas.document import (
    DocumentDetailResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
)
from app.services.ingestion import IngestionService

router = APIRouter(prefix="/api/v1", tags=["Documents Ingestion"])


def get_ingestion_service() -> IngestionService:
    """Dependency injector for IngestionService."""
    return IngestionService()


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and chunk document",
    description="Accepts document metadata and body text, partitions the document into chunks, and returns chunk details.",
    responses={
        201: {"description": "Document successfully uploaded and chunked"},
        400: {"description": "Invalid document payload"},
    },
)
async def upload_document(
    request: DocumentUploadRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentUploadResponse:
    """Uploads document content, chunks text, and stores document metadata."""
    if not request.title:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document title is required",
        )
    return service.process_document(request)


@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get document details by ID",
    description="Retrieves metadata, processing status, and text chunks of a specific ingested document.",
    responses={
        200: {"description": "Document details retrieved successfully"},
        404: {"description": "Document not found"},
    },
)
async def get_document(
    document_id: str = Path(..., description="Unique document ID", examples=["doc_123456789"]),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentDetailResponse:
    """Fetches document metadata and generated chunk details."""
    if not document_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document ID cannot be empty",
        )
    return service.get_document(document_id)
