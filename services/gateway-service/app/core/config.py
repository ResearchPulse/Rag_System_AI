from functools import lru_cache
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover
    from pydantic import BaseModel as BaseSettings  # type: ignore
    SettingsConfigDict = dict  # type: ignore


class Settings(BaseSettings):
    """Configuration settings for Gateway Service."""
    SERVICE_NAME: str = "gateway-service"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8005
    HOST: str = "0.0.0.0"
    LOG_LEVEL: str = "INFO"

    # Microservices Endpoints
    INGESTION_SERVICE_URL: str = "http://localhost:8001"
    EMBEDDING_SERVICE_URL: str = "http://localhost:8002"
    RETRIEVAL_SERVICE_URL: str = "http://localhost:8003"
    GENERATION_SERVICE_URL: str = "http://localhost:8004"

    REQUEST_TIMEOUT_SECONDS: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Provides cached instance of service settings."""
    return Settings()
