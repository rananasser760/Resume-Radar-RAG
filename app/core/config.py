from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────
    APP_NAME: str = "ResumeRadar"
    DEBUG: bool = False

    # ── Embedding ─────────────────────────────────────
    # Options: huggingface | openai | ollama
    EMBEDDING_PROVIDER: str = "huggingface"
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # ── LLM ───────────────────────────────────────────
    # Options: ollama | openrouter | groq
    LLM_PROVIDER: str = "groq"

    # Ollama
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_MODEL: str = "mistral"

    # OpenRouter
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: str = "mistralai/mistral-7b-instruct"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Groq
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # ── ChromaDB ──────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "resumes"

    # ── Chunking ──────────────────────────────────────
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # ── Retrieval ─────────────────────────────────────
    TOP_K: int = 5

    # ── Data ──────────────────────────────────────────
    RAW_DATA_DIR: str = "./raw_data"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
