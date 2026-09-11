from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Schema representing an individual text chunk from an ingested document."""
    chunk_id: str = Field(..., description="Unique chunk identifier", examples=["chk_102030"])
    chunk_index: int = Field(..., description="Sequential index of the chunk", examples=[0])
    content: str = Field(..., description="Textual body of chunk", examples=["Overview of scientometric methods."])
    character_count: int = Field(..., description="Length of chunk in characters", examples=[33])
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with chunk")


class DocumentUploadRequest(BaseModel):
    """Request schema for uploading and processing a document."""
    title: str = Field(..., description="Title or file name", examples=["Scientometrics_Review_2026.pdf"])
    content: str = Field(..., description="Raw text content to be chunked", examples=["Full paper text body..."])
    file_type: str = Field(default="txt", description="File extension/type", examples=["pdf"])
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata tags, author, year", examples=[{"authors": ["Nguyen Van A"], "year": 2026}])
    chunk_size: Optional[int] = Field(default=500, description="Target chunk character size", examples=[500])
    chunk_overlap: Optional[int] = Field(default=50, description="Chunk overlap character size", examples=[50])


class DocumentUploadResponse(BaseModel):
    """Response returned after a document has been accepted and chunked."""
    document_id: str = Field(..., description="Generated document ID", examples=["doc_998877"])
    title: str = Field(..., description="Title of document", examples=["Scientometrics_Review_2026.pdf"])
    total_chunks: int = Field(..., description="Count of chunks generated", examples=[4])
    status: str = Field(default="processed", examples=["processed"])
    chunks: List[DocumentChunk] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentDetailResponse(BaseModel):
    """Details and chunks of a stored document."""
    document_id: str = Field(..., examples=["doc_998877"])
    title: str = Field(..., examples=["Scientometrics_Review_2026.pdf"])
    file_type: str = Field(..., examples=["pdf"])
    total_chunks: int = Field(..., examples=[4])
    status: str = Field(..., examples=["processed"])
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunks: List[DocumentChunk] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
