from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
import shutil
import tempfile

# Import the ResumeRadar RAG Pipeline
from app.services.rag_pipeline import RAGPipeline

# Initialize FastAPI
app = FastAPI(title="ResumeRadar API", version="1.0.0")

# Initialize the RAG Pipeline singleton
print("[Startup] 🔥 Warming up ResumeRadar API...")
rag = RAGPipeline()

# --- Frontend Serving --------------------------------------------------------

os.makedirs("app/static", exist_ok=True)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_frontend():
    """Serves the Vanilla JS frontend interface."""
    try:
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>UI Not Found</h1><p>Please ensure app/static/index.html exists.</p>"

# --- Pydantic Data Models ----------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

class IngestRequest(BaseModel):
    dir_path: str = "raw_data"
    clear_first: bool = False

class SwitchLLMRequest(BaseModel):
    provider: str
    model: Optional[str] = None

# --- API Routes --------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    """Basic health check for Docker/Uvicorn."""
    return {"status": "healthy", "stats": rag.stats()}

@app.post("/api/v1/ingest", tags=["RAG"])
async def ingest_documents(req: IngestRequest):
    """Processes PDFs from the specified directory into ChromaDB."""
    try:
        result = rag.ingest(dir_path=req.dir_path, clear_first=req.clear_first)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/api/v1/query", tags=["RAG"])
async def query_pipeline(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        result = rag.query(
            user_query=req.query,
            top_k=req.top_k,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()  # ← السطر ده بس اللي اتضاف
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
@app.post("/api/v1/switch-llm", tags=["System"])
async def switch_llm(req: SwitchLLMRequest):
    """Hot-swaps the LLM provider at runtime (e.g., from Groq to Ollama)."""
    try:
        active_llm = rag.switch_llm(provider=req.provider, model=req.model)
        return {
            "status": "success",
            "active_provider": active_llm.provider_name,
            "active_model": active_llm.model_name
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/v1/clear", tags=["System"])
async def clear_database():
    """Wipes the ChromaDB collection entirely to start fresh."""
    try:
        rag.vector_store.clear()
        return {"status": "success", "message": "Vector database cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear DB: {str(e)}")

# ── Upload single file ─────────────────────────────────────────────────────
@app.post("/api/v1/upload", tags=["RAG"])
async def upload_single_file(file: UploadFile = File(...)):
    """
    Upload a single file (.pdf, .docx, .html) and index it into ChromaDB immediately.
    Supported formats: .pdf, .docx, .html, .htm
    """
    supported = {".pdf", ".docx", ".html", ".htm"}
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(supported))}"
        )
    named_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        named_path = os.path.join(os.path.dirname(tmp_path), file.filename)
        os.rename(tmp_path, named_path)
        result = rag.ingest_single_file(named_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        if named_path and os.path.exists(named_path):
            try:
                os.remove(named_path)
            except Exception:
                pass


# ✅ NEW — GET /api/v1/candidates
@app.get("/api/v1/candidates", tags=["RAG"])
async def get_candidates():
    """Returns all unique candidates grouped by filename from ChromaDB."""
    try:
        all_metadata = rag.vector_store.get_all_metadata()

        candidates = {}
        for metadata in all_metadata:
            fname = metadata.get("filename", "unknown")
            if fname not in candidates:
                candidates[fname] = {
                    "filename": fname,
                    "language": metadata.get("language", "unknown"),
                    "arabic_ratio": float(metadata.get("arabic_ratio", 0)),
                    "chunks": 0
                }
            candidates[fname]["chunks"] += 1

        return {"candidates": list(candidates.values())}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch candidates: {str(e)}")