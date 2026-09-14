from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_context_memory_service
from app.modules.generation.context_memory.schemas import (
    ContextMemoryState,
    ResetContextMemoryResponse,
    UpdateContextMemoryRequest,
)
from app.modules.generation.context_memory.service import ContextMemoryService

router = APIRouter(prefix="/chat/context", tags=["Context Memory"])


@router.get(
    "",
    response_model=ContextMemoryState,
    status_code=status.HTTP_200_OK,
    summary="Get Context Memory",
    description="Retrieve the active conversation working memory and entities for a user/project session.",
)
async def get_context_memory(
    project_id: Optional[int] = Query(default=None, description="Project ID"),
    user_id: Optional[str] = Query(default=None, description="User UUID"),
    service: ContextMemoryService = Depends(get_context_memory_service),
) -> ContextMemoryState:
    return service.get_memory(project_id, user_id)


@router.post(
    "/update",
    response_model=ContextMemoryState,
    status_code=status.HTTP_200_OK,
    summary="Update Context Memory",
    description="Manually update active focus topic, target publication year, or referenced entities.",
)
async def update_context_memory(
    payload: UpdateContextMemoryRequest,
    service: ContextMemoryService = Depends(get_context_memory_service),
) -> ContextMemoryState:
    return service.update_memory(payload)


@router.post(
    "/reset",
    response_model=ResetContextMemoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Reset Context Memory",
    description="Clear active working memory and recent conversational turns for a fresh dialogue session.",
)
async def reset_context_memory(
    project_id: Optional[int] = Query(default=None, description="Project ID"),
    user_id: Optional[str] = Query(default=None, description="User UUID"),
    service: ContextMemoryService = Depends(get_context_memory_service),
) -> ResetContextMemoryResponse:
    return service.reset_memory(project_id, user_id)
