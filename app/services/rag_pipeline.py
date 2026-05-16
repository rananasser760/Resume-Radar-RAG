"""
RAG Pipeline — Orchestrates all phases.
Phase 1: Parse → Chunk → Embed → Store
Phase 3: Query (Multi-Query Expansion) → Retrieve → Generate (Metadata Injected)
"""
from typing import List, Dict, Any, Optional
from app.services.parser import PDFParser
from app.services.chunker import ChunkingService
from app.services.embedder import EmbedderFactory
from app.services.vector_store import VectorStore
from app.services.llm import LLMFactory
from app.core.config import settings


class RAGPipeline:
    """Orchestrator for the ResumeRadar recruitment platform."""

    def __init__(self):
        print("[RAGPipeline] 🚀 Initializing Orchestrator...")
        self.parser = PDFParser()
        self.chunker = ChunkingService()
        self.embedder = EmbedderFactory.create(settings.EMBEDDING_PROVIDER)
        self.vector_store = VectorStore(embedder=self.embedder)
        self.llm_factory = LLMFactory() 
        
        print(
            f"[RAGPipeline] Ready | "
            f"Embedder: {self.embedder.model_name} | "
            f"Active LLM: {self.active_llm.provider_name}/{self.active_llm.model_name}"
        )

    # ── LLM Management ────────────────────────────────

    def switch_llm(self, provider: str, model: Optional[str] = None):
        """Allows hot-swapping between Groq and local TinyLlama."""
        return self.llm_factory.switch(provider, model)

    @property
    def active_llm(self):
        return self.llm_factory.current

    # ── Ingestion ─────────────────────────────────────

    def ingest_single_file(self, file_path: str) -> Dict[str, Any]:
        """Parses a single file (.pdf, .docx, .html) and adds it to the vector database."""
        print(f"\n[RAGPipeline] 📄 Ingesting single file: {file_path}")
        try:
            doc = self.parser.parse_file(file_path)
        except ValueError as e:
            return {"files_processed": 0, "message": str(e)}

        chunks = self.chunker.chunk_document(doc)
        if not chunks:
            return {"files_processed": 0, "message": f"No text extracted from {doc['filename']}."}

        stored_count = self.vector_store.add_chunks(chunks)
        return {
            "files_processed": 1,
            "chunks_stored": stored_count,
            "message": f"Successfully indexed {doc['filename']}.",
            "file_details": [{
                "filename": doc["filename"],
                "language": doc["language"],
                "arabic_ratio": doc["arabic_ratio"],
                "chunks": len(chunks),
            }],
        }

    def ingest(self, dir_path: str, clear_first: bool = False) -> Dict[str, Any]:
        """Processes raw PDFs into the searchable vector database."""
        if clear_first:
            self.vector_store.clear()

        print(f"\n[RAGPipeline] 📂 Starting Ingestion: {dir_path}")
        docs = self.parser.parse_directory(dir_path)
        
        if not docs:
            return {
                "files_processed": 0, 
                "message": "No valid PDFs found for processing.",
            }

        chunks = self.chunker.chunk_documents(docs)
        stored_count = self.vector_store.add_chunks(chunks)

        file_details = [
            {
                "filename": d["filename"],
                "language": d["language"],
                "arabic_ratio": d["arabic_ratio"],
                "chunks": len([c for c in chunks if c["metadata"]["filename"] == d["filename"]]),
            }
            for d in docs
        ]

        return {
            "files_processed": len(docs),
            "chunks_stored": stored_count,
            "message": f"Successfully indexed {len(docs)} resumes.",
            "file_details": file_details,
        }

    # ── Intelligent Querying ──────────────────────────

    def query(
        self,
        user_query: str,
        top_k: int = settings.TOP_K,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handles the RAG cycle with Multi-Query Expansion for Arabic/English cross-retrieval.
        """
        print(f"\n[RAGPipeline] 🔍 Processing Query: {user_query!r}")

        # 1. Hot-swap LLM if requested (e.g., switching to TinyLlama for local testing)
        if llm_provider:
            self.switch_llm(llm_provider, llm_model)

        # 2. Multi-Query Expansion: Bridge the English-to-Arabic gap
        # This solves the issue where English queries miss purely Arabic CVs.
        search_queries = [user_query]
        if "arabic" in user_query.lower() or "عربي" in user_query:
            search_queries.append("سيرة ذاتية باللغة العربية") # "Arabic Resume"
            search_queries.append("الخبرة العملية") # "Work Experience"

        # 3. Retrieve and Deduplicate
        raw_results = []
        for q in search_queries:
            raw_results.extend(self.vector_store.query(q, top_k=top_k))
        
        # Keep unique chunks based on ID, favoring the highest similarity score
        unique_chunks = {}
        for res in raw_results:
            cid = res["chunk_id"]
            if cid not in unique_chunks or res["score"] > unique_chunks[cid]["score"]:
                unique_chunks[cid] = res
        
        # Sort and take the top results
        sorted_chunks = sorted(unique_chunks.values(), key=lambda x: x["score"], reverse=True)[:top_k]

        if not sorted_chunks:
            return {"query": user_query, "answer": "No relevant CV data found.", "chunks_used": 0}

        # 4. Contextual Injection (Metadata Baking)
        # We explicitly tell the LLM the language of each chunk to avoid hallucinations.
        context_blocks = []
        for chunk in sorted_chunks:
            meta = chunk["metadata"]
            block = (
                f"--- DOCUMENT: {meta.get('filename')} ---\n"
                f"METADATA: [Language: {meta.get('language')}, Arabic Ratio: {meta.get('arabic_ratio')}]\n"
                f"CONTENT: {chunk['content']}\n"
            )
            context_blocks.append(block)

        # 5. Generate Answer
        answer = self.active_llm.generate(query=user_query, context_chunks=context_blocks)
        
        return {
            "query": user_query,
            "answer": answer,
            "retrieved_chunks": sorted_chunks,
            "llm_provider": self.active_llm.provider_name,
            "llm_model": self.active_llm.model_name,
            "chunks_used": len(sorted_chunks),
        }

    # ── System Stats ──────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Returns diagnostic information for the ResumeRadar dashboard."""
        return {
            "total_chunks_in_db": self.vector_store.count(),
            "embedding_config": f"{self.embedder.provider_name}/{self.embedder.model_name}",
            "active_llm": f"{self.active_llm.provider_name}/{self.active_llm.model_name}",
            "available_providers": LLMFactory.available(),
        }