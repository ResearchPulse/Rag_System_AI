from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.shared.schemas import HealthCheckResponse

settings = get_settings()

app = FastAPI(
    title="Rag_System_AI - Modular Monolith",
    description=(
        "Scientific Journal Publication Trend Tracking System.\n\n"
        "Unified Modular Monolith RAG Pipeline covering all 12 Steps across 4 Phases:\n"
        "- Phase 1: Ingestion (Steps 1-4: Data Sources, Loading, Meaningful Chunking, Metadata Extraction)\n"
        "- Phase 2: Indexing (Steps 5-6: Embeddings, Vector Database)\n"
        "- Phase 3: Retrieval (Steps 7-9: Query Rewriting, Hybrid Search, Cross-Encoder Reranking)\n"
        "- Phase 4: Generation & Evaluation (Steps 10-12: Context Assembly, LLM Generation, Evaluation)"
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="Application Health Check",
    description="Returns service health status and environment.",
)
async def health_check() -> HealthCheckResponse:
    return HealthCheckResponse(
        status="healthy",
        version="0.1.0",
        environment=settings.APP_ENV,
    )


@app.get(
    "/",
    tags=["System"],
    summary="Root Welcome & Sitemap",
)
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "architecture": "Modular Monolith",
        "docs": "/docs",
        "health": "/health",
        "api_v1_prefix": "/api/v1",
    }


app.include_router(api_v1_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
