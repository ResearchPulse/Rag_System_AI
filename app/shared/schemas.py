from datetime import datetime, timezone
from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class BaseResponse(BaseModel, Generic[T]):
    """Standard generic API response wrapper."""
    success: bool = Field(default=True, description="Indicates if the operation was successful")
    message: str = Field(default="Operation completed successfully", description="Status message")
    data: Optional[T] = Field(default=None, description="Payload data")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp in UTC")


class ErrorDetail(BaseModel):
    """Details about an error."""
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable explanation")
    details: Optional[Any] = Field(default=None, description="Detailed trace or context")


class ErrorResponse(BaseModel):
    """Standard error response wrapper."""
    success: bool = Field(default=False, description="Always false for error responses")
    error: ErrorDetail = Field(..., description="Error detail information")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp in UTC")


class HealthCheckResponse(BaseModel):
    """Standard system health status response."""
    status: str = Field(default="healthy", examples=["healthy"])
    version: str = Field(default="0.1.0", examples=["0.1.0"])
    environment: str = Field(default="development", examples=["development"])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
