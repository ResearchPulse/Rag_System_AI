"""Application dependency injectors for Modular Monolith services."""
from functools import lru_cache
from app.modules.generation.service import GenerationService
from app.modules.indexing.service import IndexingService
from app.modules.ingestion.service import IngestionService
from app.modules.retrieval.service import RetrievalService
from app.modules.chat_history.service import ChatHistoryService
from app.modules.generation.context_memory.service import ContextMemoryService


@lru_cache()
def get_ingestion_service() -> IngestionService:
    return IngestionService()


@lru_cache()
def get_indexing_service() -> IndexingService:
    return IndexingService()


@lru_cache()
def get_retrieval_service() -> RetrievalService:
    return RetrievalService()


@lru_cache()
def get_generation_service() -> GenerationService:
    return GenerationService()


@lru_cache()
def get_chat_history_service() -> ChatHistoryService:
    return ChatHistoryService()


@lru_cache()
def get_context_memory_service() -> ContextMemoryService:
    return ContextMemoryService()


