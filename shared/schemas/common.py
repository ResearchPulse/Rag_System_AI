from datetime import datetime, timezone
from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class BaseResponse(BaseModel, Generic[T]):
    """Standard generic API response wrapper."""
    success: bool = Field(default=True, description="Indicates if the operation was successful")
    message: str = Field(default="Operation completed successfully", description="Status message")
    data: Optional[T] = Field(default=None, description="Payload data")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of the response in UTC")


class ErrorDetail(BaseModel):
    """Details about an error."""
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error explanation")
    details: Optional[Any] = Field(default=None, description="Detailed trace or context of the error")


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = Field(default=False, description="Always false for error responses")
    error: ErrorDetail = Field(..., description="Error detail information")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of the response in UTC")


class HealthCheckResponse(BaseModel):
    """Standard health check response."""
    service: str = Field(..., description="Service identifier name")
    status: str = Field(default="healthy", description="Current service health status")
    version: str = Field(..., description="Service semantic version")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp in UTC")

