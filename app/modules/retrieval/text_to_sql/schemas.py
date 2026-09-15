"""Pydantic schemas for the Text-to-SQL (NL2SQL) Engine."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SQLGenerationRequest(BaseModel):
    query: str = Field(..., description="User's natural language research question")
    project_id: Optional[int] = Field(None, description="Active project ID for scoping")
    dialect: str = Field(default="postgresql", description="SQL dialect, defaults to postgresql")


class SQLValidationResult(BaseModel):
    is_valid: bool = Field(..., description="Whether the generated SQL is safe and valid")
    sanitized_sql: Optional[str] = Field(None, description="Sanitized executable SQL query")
    error_message: Optional[str] = Field(None, description="Rejection reason if SQL is invalid or unsafe")


class SQLExecutionResult(BaseModel):
    sql: str = Field(..., description="The executed SQL query")
    columns: List[str] = Field(default_factory=list, description="Column names returned")
    rows: List[List[Any]] = Field(default_factory=list, description="Row values")
    row_count: int = Field(default=0, description="Total number of rows returned")
    formatted_markdown: str = Field(default="", description="Markdown representation of results")
    success: bool = Field(default=True, description="Whether execution succeeded")
    error: Optional[str] = Field(default=None, description="Execution error if any")
    execution_time_ms: float = Field(default=0.0, description="Query execution time in milliseconds")
