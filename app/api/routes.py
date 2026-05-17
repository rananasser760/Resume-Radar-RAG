"""
API Routes — MVC Controller Layer
All endpoints under /api/v1/
"""
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, RetrievedChunk,
    CollectionStatsResponse,
    SwitchLLMRequest, SwitchLLMResponse,
)
from app.services.rag_pipeline import RAGPipeline
from app.core.config import settings

router = APIRouter()

# ── Singleton pipeline ────────────────────────────────
_pipeline: RAGPipeline = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ─────────────────────────────────────────────────────
#  POST /ingest
# ─────────────────────────────────────────────────────
@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Ingest all PDF resumes from ./raw_data/",
)
def ingest(req: IngestRequest = IngestRequest()):
    """
    Triggers the full ingestion pipeline:
    1. Parse all PDFs in `./raw_data/` (supports Arabic + English)
    2. Chunk with RecursiveCharacterTextSplitter (500 chars / 50 overlap)
    3. Embed with multilingual model
    4. Store in ChromaDB with full metadata

    Set `clear_before_ingest=true` to wipe and re-index from scratch.
    """
    try:
        p = get_pipeline()
        result = p.ingest(
            dir_path=settings.RAW_DATA_DIR,
            clear_first=req.clear_before_ingest,
        )
        return IngestResponse(status="success", **result)
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ─────────────────────────────────────────────────────
#  POST /query
# ─────────────────────────────────────────────────────
@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["Query"],
    summary="RAG query — English or Arabic",
)
def query(req: QueryRequest):
    """
    Full RAG pipeline per request:
    1. Embed user query
    2. Retrieve top-k chunks from ChromaDB (cosine similarity)
    3. Inject context into LLM prompt
    4. Return answer + source chunks

    Override LLM per-request via `llm_provider` + `llm_model` fields.

    **Arabic example:**
    ```json
    { "query": "من لديه خبرة في تعلم الآلة؟", "top_k": 5 }
    ```
    """
    try:
        p = get_pipeline()
        result = p.query(
            user_query=req.query,
            top_k=req.top_k,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
        )
        chunks = [
            RetrievedChunk(
                content=c["content"],
                source=c["source"],
                chunk_id=c["chunk_id"],
                score=c["score"],
                metadata=c["metadata"],
            )
            for c in result["retrieved_chunks"]
        ]
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            retrieved_chunks=chunks,
            llm_provider=result["llm_provider"],
            llm_model=result["llm_model"],
            chunks_used=result["chunks_used"],
        )
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ─────────────────────────────────────────────────────
#  POST /switch-llm
# ─────────────────────────────────────────────────────
@router.post(
    "/switch-llm",
    response_model=SwitchLLMResponse,
    tags=["LLM Control"],
    summary="Hot-swap the active LLM provider at runtime",
)
def switch_llm(req: SwitchLLMRequest):
    """
    Switch LLM provider without restarting the server.

    **Providers:**
    - `ollama` — local Mistral/LLaMA via Ollama server
    - `openrouter` — cloud models (Mistral, Claude, GPT-4…)
    - `groq` — ultra-fast Groq inference (LLaMA3, Mixtral…)

    **Example — switch to Groq Mixtral:**
    ```json
    { "provider": "groq", "model": "mixtral-8x7b-32768" }
    ```

    **Example — switch back to local Ollama:**
    ```json
    { "provider": "ollama", "model": "mistral" }
    ```
    """
    try:
        p = get_pipeline()
        llm = p.switch_llm(req.provider, req.model)
        return SwitchLLMResponse(
            status="success",
            provider=llm.provider_name,
            model=llm.model_name,
            message=f"Switched to {llm.provider_name}/{llm.model_name}",
        )
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ─────────────────────────────────────────────────────
#  GET /stats
# ─────────────────────────────────────────────────────
@router.get(
    "/stats",
    response_model=CollectionStatsResponse,
    tags=["Utilities"],
    summary="Vector DB + model configuration stats",
)
def stats():
    try:
        return CollectionStatsResponse(**get_pipeline().stats())
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ─────────────────────────────────────────────────────
#  DELETE /clear
# ─────────────────────────────────────────────────────
@router.delete(
    "/clear",
    tags=["Utilities"],
    summary="Clear all vectors from ChromaDB",
)
def clear():
    """⚠️ Deletes all stored embeddings. Re-run /ingest afterwards."""
    try:
        get_pipeline().vector_store.clear()
        return {"status": "success", "message": "Collection cleared."}
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ─────────────────────────────────────────────────────
#  GET /providers
# ─────────────────────────────────────────────────────
@router.get(
    "/providers",
    tags=["LLM Control"],
    summary="List all available LLM providers and their default models",
)
def providers():
    from app.services.llm import LLMFactory
    from app.services.embedder import EmbedderFactory
    p = get_pipeline()
    return {
        "active_llm": {
            "provider": p.active_llm.provider_name,
            "model": p.active_llm.model_name,
        },
        "llm_providers": LLMFactory.available(),
        "llm_default_models": LLMFactory.default_models(),
        "embedding_providers": EmbedderFactory.available(),
        "active_embedding": {
            "provider": p.embedder.provider_name,
            "model": p.embedder.model_name,
        },
    }
