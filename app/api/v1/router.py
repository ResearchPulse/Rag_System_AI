from fastapi import APIRouter
from app.api.v1.endpoints.embedding import router as embedding_router
from app.api.v1.endpoints.generation import router as generation_router
from app.api.v1.endpoints.ingestion import router as ingestion_router
from app.api.v1.endpoints.rag import router as rag_router
from app.api.v1.endpoints.retrieval import router as retrieval_router

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(ingestion_router)
api_v1_router.include_router(embedding_router)
api_v1_router.include_router(retrieval_router)
api_v1_router.include_router(generation_router)
api_v1_router.include_router(rag_router)
