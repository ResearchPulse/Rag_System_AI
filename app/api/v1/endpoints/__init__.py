from app.api.v1.endpoints.embedding import router as embedding_router
from app.api.v1.endpoints.generation import router as generation_router
from app.api.v1.endpoints.ingestion import router as ingestion_router
from app.api.v1.endpoints.rag import router as rag_router
from app.api.v1.endpoints.retrieval import router as retrieval_router

__all__ = [
    "embedding_router",
    "generation_router",
    "ingestion_router",
    "rag_router",
    "retrieval_router",
]
