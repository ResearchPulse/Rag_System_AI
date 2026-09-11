import uuid
from datetime import datetime, timezone
from typing import List
from app.modules.ingestion.schemas import (
    DocumentChunk,
    DocumentDetailResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
)


class IngestionService:
    """Service orchestrating Phase 1: Ingestion (Steps 1 to 4)."""

    def process_document(self, request: DocumentUploadRequest) -> DocumentUploadResponse:
        """Processes document text and produces mock semantic chunk partitions."""
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        content = request.content or f"Sample content for {request.title}"
        chunk_size = request.chunk_size or 500
        chunk_overlap = request.chunk_overlap or 50

        step = max(chunk_size - chunk_overlap, 1)
        chunks: List[DocumentChunk] = []

        for idx, i in enumerate(range(0, len(content), step)):
            chunk_slice = content[i : i + chunk_size]
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chk_{uuid.uuid4().hex[:6]}",
                    chunk_index=idx,
                    content=chunk_slice,
                    character_count=len(chunk_slice),
                    metadata={
                        **request.metadata,
                        "document_id": doc_id,
                        "title": request.title,
                        "chunk_index": idx,
                    },
                )
            )

        if not chunks:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chk_{uuid.uuid4().hex[:6]}",
                    chunk_index=0,
                    content=content,
                    character_count=len(content),
                    metadata={"document_id": doc_id, "title": request.title},
                )
            )

        return DocumentUploadResponse(
            document_id=doc_id,
            title=request.title,
            total_chunks=len(chunks),
            status="processed",
            chunks=chunks,
            created_at=datetime.now(timezone.utc),
        )

    def get_document(self, document_id: str) -> DocumentDetailResponse:
        """Retrieves details and mock chunks for a given document id."""
        mock_chunks = [
            DocumentChunk(
                chunk_id=f"chk_{document_id}_01",
                chunk_index=0,
                content="Introduction to scientific publication trend tracking and AI methods.",
                character_count=67,
                metadata={"section": "Introduction", "document_id": document_id},
            ),
            DocumentChunk(
                chunk_id=f"chk_{document_id}_02",
                chunk_index=1,
                content="Methodology: Modular Monolith architecture for scholarly RAG.",
                character_count=64,
                metadata={"section": "Methodology", "document_id": document_id},
            ),
        ]
        return DocumentDetailResponse(
            document_id=document_id,
            title="Scientific Publication Trend Tracking System Overview.pdf",
            file_type="pdf",
            total_chunks=len(mock_chunks),
            status="processed",
            metadata={"domain": "AI / Scientometrics", "status": "indexed"},
            chunks=mock_chunks,
            created_at=datetime.now(timezone.utc),
        )
