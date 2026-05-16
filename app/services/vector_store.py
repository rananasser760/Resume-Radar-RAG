"""
Vector Store — ChromaDB persistent wrapper.
Cosine similarity, upsert-safe, full metadata support.
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any
from app.core.config import settings
from app.services.embedder import BaseEmbedder


class VectorStore:

    def __init__(self, embedder: BaseEmbedder, collection_name: str = None):
        self.embedder = embedder
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME

        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._col = self._get_or_create()
        print(f"[VectorStore] ✅ '{self.collection_name}' | {self._col.count()} chunks")

    def _get_or_create(self):
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ─────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Embed + upsert chunks. Idempotent — safe to re-run."""
        if not chunks:
            return 0
        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metas = [c["metadata"] for c in chunks]

        # Batch embed (handles large sets efficiently)
        embeddings = self.embedder.embed_texts(texts)

        self._col.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metas,
        )
        print(f"[VectorStore] 💾 Upserted {len(chunks)} chunks.")
        return len(chunks)

    # ── Read ──────────────────────────────────────────

    def query(self, query: str, top_k: int = settings.TOP_K) -> List[Dict[str, Any]]:
        """Embed query → cosine search → return top-k chunks."""
        total = self._col.count()
        if total == 0:
            return []

        k = min(top_k, total)
        q_emb = self.embedder.embed_query(query)

        res = self._col.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            results.append({
                "content": doc,
                "source": meta.get("filename", "unknown"),
                "chunk_id": meta.get("chunk_id", ""),
                "score": round(1.0 - dist, 4),   # cosine distance → similarity
                "metadata": meta,
            })
        return results

    # ── Utilities ─────────────────────────────────────

    def count(self) -> int:
        return self._col.count()

    def clear(self):
        self._client.delete_collection(self.collection_name)
        self._col = self._get_or_create()
        print(f"[VectorStore] 🗑️  Collection '{self.collection_name}' cleared.")

    # في vector_store.py — جوه class VectorStore
    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Returns all stored metadata entries."""
        results = self._col.get(include=["metadatas"])
        return results["metadatas"]