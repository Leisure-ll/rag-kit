from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Kit"
    index_dir: Path = Path("data/index")
    chunk_size: int = 650
    chunk_overlap: int = 120
    top_k: int = 5
    vector_weight: float = 0.65
    bm25_weight: float = 0.35
    embedding_backend: str = "hashing"
    sentence_transformer_model: str = "BAAI/bge-small-zh-v1.5"

    llm_api_key: Optional[str] = None
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_timeout: float = 60.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
