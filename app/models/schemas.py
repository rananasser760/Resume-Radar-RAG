from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ── Ingestion ─────────────────────────────────────────

class IngestRequest(BaseModel):
    collection_name: Optional[str] = None
    clear_before_ingest: bool = False  # wipe collection first


class IngestResponse(BaseModel):
    status: str
    files_processed: int
    chunks_stored: int
    skipped: int = 0
    message: str
    file_details: List[Dict[str, Any]] = []


# ── Query ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query (Arabic or English)")
    top_k: Optional[int] = Field(5, ge=1, le=20)
    llm_provider: Optional[str] = None   # runtime override: ollama | openrouter | groq
    llm_model: Optional[str] = None      # runtime override of model name


class RetrievedChunk(BaseModel):
    content: str
    source: str
    chunk_id: str
    score: float
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    llm_provider: str
    llm_model: str
    chunks_used: int


# ── Stats ─────────────────────────────────────────────

class CollectionStatsResponse(BaseModel):
    collection_name: str
    total_chunks: int
    embedding_model: str
    embedding_provider: str
    default_llm_provider: str
    default_llm_model: str
    available_llm_providers: List[str]
    available_embedding_providers: List[str]


# ── LLM Switch ────────────────────────────────────────

class SwitchLLMRequest(BaseModel):
    provider: str = Field(..., description="ollama | openrouter | groq")
    model: Optional[str] = None  # if None, uses provider default


class SwitchLLMResponse(BaseModel):
    status: str
    provider: str
    model: str
    message: str


# ── Error ─────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
