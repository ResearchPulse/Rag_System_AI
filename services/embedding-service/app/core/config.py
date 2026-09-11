from functools import lru_cache
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover
    from pydantic import BaseModel as BaseSettings  # type: ignore
    SettingsConfigDict = dict  # type: ignore


class Settings(BaseSettings):
    """Configuration settings for Embedding Service."""
    SERVICE_NAME: str = "embedding-service"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8002
    HOST: str = "0.0.0.0"
    LOG_LEVEL: str = "INFO"

    # Embedding Model Settings
    DEFAULT_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    BATCH_SIZE: int = 32

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
