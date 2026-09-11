from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app.api.v1.routes import router as v1_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Retrieval Service - Rag_System_AI",
    description="Microservice for vector semantic search, knowledge base lookup, and cross-encoder reranking.",
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


class HealthResponse(BaseModel):
    service: str = Field(..., example="retrieval-service")
    status: str = Field(default="healthy", example="healthy")
    version: str = Field(default="0.1.0", example="0.1.0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Returns the operational status of the Retrieval Service.",
)
async def health_check() -> HealthResponse:
    """Verifies that the Retrieval Service is operational."""
    return HealthResponse(
        service=settings.SERVICE_NAME,
        status="healthy",
        version="0.1.0",
    )


app.include_router(v1_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
