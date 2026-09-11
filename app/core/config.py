from functools import lru_cache
from typing import Optional
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover
    from pydantic import BaseModel as BaseSettings  # type: ignore
    SettingsConfigDict = dict  # type: ignore


class Settings(BaseSettings):
    """Unified configuration for the Rag_System_AI Modular Monolith."""
    PROJECT_NAME: str = "Rag_System_AI"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    LOG_LEVEL: str = "INFO"

    # Phase 1: Ingestion
    UPLOAD_DIR: str = "./uploads"
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50

    # Phase 2: Indexing (PgVector & Neo4j Graph DB)
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    VECTOR_DB_TYPE: str = "pgvector"

    # PostgreSQL / pgvector Storage (managed via pgAdmin)
    POSTGRES_HOST: str = "100.121.61.95"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "researchpulse"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres123"
    POSTGRES_SCHEMA: str = "public"
    POSTGRES_URL: Optional[str] = None

    # Neo4j Knowledge Graph Storage
    NEO4J_URI: str = "bolt://100.121.61.95:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "neo4j123"
    NEO4J_DATABASE: str = "neo4j"

    # Phase 3: Retrieval
    DEFAULT_TOP_K: int = 5
    DEFAULT_SCORE_THRESHOLD: float = 0.5
    ENABLE_RERANKING: bool = True

    # Phase 4: Generation
    LLM_PROVIDER: str = "mock"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1024
    OPENAI_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Provides cached instance of application settings."""
    return Settings()
