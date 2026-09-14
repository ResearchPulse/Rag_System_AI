from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_chat_history_service
from app.modules.chat_history.schemas import (
    ChatMessageCreate,
    ChatMessageItem,
    ChatHistoryResponse,
    DeleteHistoryResponse,
)
from app.modules.chat_history.service import ChatHistoryService

router = APIRouter(prefix="/chat", tags=["Chat History"])


@router.get(
    "/history",
    response_model=ChatHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Chat History",
    description="Retrieve paginated conversation history filtered by project_id and/or user_id.",
)
async def get_chat_history(
    project_id: Optional[int] = Query(default=None, description="Optional project ID filter"),
    user_id: Optional[str] = Query(default=None, description="Optional user UUID filter"),
    limit: int = Query(default=50, ge=1, le=100, description="Max messages to return (1-100)"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    order: str = Query(default="asc", pattern="^(asc|desc|ASC|DESC)$", description="Sort order by time ('asc' or 'desc')"),
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> ChatHistoryResponse:
    return service.get_history(
        project_id=project_id,
        user_id=user_id,
        limit=limit,
        offset=offset,
        order=order,
    )


@router.post(
    "/messages",
    response_model=ChatMessageItem,
    status_code=status.HTTP_201_CREATED,
    summary="Create Chat Message",
    description="Manually create a chat message record (USER, ASSISTANT, or SYSTEM) in conversation history.",
)
async def create_chat_message(
    payload: ChatMessageCreate,
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> ChatMessageItem:
    if not payload.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content cannot be empty.",
        )
    return service.save_message(payload)


@router.get(
    "/messages/{message_id}",
    response_model=ChatMessageItem,
    status_code=status.HTTP_200_OK,
    summary="Get Message Detail",
    description="Retrieve details of a single message by ID.",
)
async def get_message_detail(
    message_id: int,
    project_id: Optional[int] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> ChatMessageItem:
    msg = service.get_message(message_id, project_id, user_id)
    if not msg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat message with ID {message_id} not found.",
        )
    return msg


@router.delete(
    "/messages/{message_id}",
    response_model=DeleteHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete Single Message",
    description="Delete a specific chat message by its ID.",
)
async def delete_chat_message(
    message_id: int,
    project_id: Optional[int] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> DeleteHistoryResponse:
    res = service.delete_message(message_id, project_id, user_id)
    if res.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Message ID {message_id} not found or already deleted.",
        )
    return res


@router.delete(
    "/history",
    response_model=DeleteHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear Chat History",
    description="Delete all conversation history for a given project_id and/or user_id.",
)
async def clear_chat_history(
    project_id: Optional[int] = Query(default=None, description="Project ID whose history to clear"),
    user_id: Optional[str] = Query(default=None, description="User UUID whose history to clear"),
    service: ChatHistoryService = Depends(get_chat_history_service),
) -> DeleteHistoryResponse:
    if project_id is None and user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 'project_id' or 'user_id' must be provided to clear chat history.",
        )
    return service.clear_history(project_id, user_id)
