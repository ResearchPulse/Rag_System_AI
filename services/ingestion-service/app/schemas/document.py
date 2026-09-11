from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Schema representing an individual text chunk from an ingested document."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk", examples=["chk_987654321"])
    chunk_index: int = Field(..., description="Sequential index of the chunk within the document", examples=[0])
    content: str = Field(..., description="Textual content of the chunk", examples=["This is the first segment of text."])
    character_count: int = Field(..., description="Total characters in this chunk", examples=[37])
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata associated with this chunk")


class DocumentUploadRequest(BaseModel):
    """Request schema for uploading and processing a document."""
    title: str = Field(..., description="Title or file name of the document", examples=["Research_Paper_2026.pdf"])
    content: str = Field(..., description="Raw text content or extracted body to be chunked", examples=["Full text of the paper..."])
    file_type: str = Field(default="txt", description="Document file extension/type (pdf, docx, txt)", examples=["pdf"])
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata such as authors, tags, journal", examples=[{"authors": ["Nguyen Van A"], "year": 2026}])
    chunk_size: Optional[int] = Field(default=500, description="Override chunk size for splitting", examples=[500])
    chunk_overlap: Optional[int] = Field(default=50, description="Override chunk overlap for splitting", examples=[50])


class DocumentUploadResponse(BaseModel):
    """Response schema returned after a document has been accepted and chunked."""
    document_id: str = Field(..., description="Generated document ID", examples=["doc_123456789"])
    title: str = Field(..., description="Title of the ingested document", examples=["Research_Paper_2026.pdf"])
    total_chunks: int = Field(..., description="Number of chunks generated", examples=[5])
    status: str = Field(default="processed", description="Current ingestion status", examples=["processed"])
    chunks: List[DocumentChunk] = Field(default_factory=list, description="List of generated chunks")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when ingested")


class DocumentDetailResponse(BaseModel):
    """Response schema providing details and chunks of a previously ingested document."""
    document_id: str = Field(..., description="Document identifier", examples=["doc_123456789"])
    title: str = Field(..., description="Title of the document", examples=["Research_Paper_2026.pdf"])
    file_type: str = Field(..., description="Type of document (pdf, txt, docx)", examples=["pdf"])
    total_chunks: int = Field(..., description="Total count of text chunks", examples=[5])
    status: str = Field(..., description="Ingestion processing status", examples=["processed"])
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Associated document metadata")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="List of chunks")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")

