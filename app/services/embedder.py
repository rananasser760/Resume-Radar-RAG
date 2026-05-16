"""
Embedding Factory Pattern — Phase 1 / Bonus 1
Providers: huggingface | openai | ollama
Switch via EMBEDDING_PROVIDER env var — zero code change.
"""
from abc import ABC, abstractmethod
from typing import List
from app.core.config import settings


# ── Abstract Base ─────────────────────────────────────

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]: ...

    @abstractmethod
    def embed_query(self, query: str) -> List[float]: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...


# ── HuggingFace (default — local, multilingual) ───────

class HuggingFaceEmbedder(BaseEmbedder):
    """
    Uses sentence-transformers locally.
    Model: paraphrase-multilingual-MiniLM-L12-v2
    Supports Arabic + English in the SAME vector space → cross-lingual retrieval.
    """
    def __init__(self, model: str = settings.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        self._model_id = model
        self._model = SentenceTransformer(model)
        print(f"[Embedder] ✅ HuggingFace model loaded: {model}")

    @property
    def model_name(self): return self._model_id
    @property
    def provider_name(self): return "huggingface"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, show_progress_bar=False, batch_size=32).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self._model.encode([query], show_progress_bar=False)[0].tolist()


# ── OpenAI ────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = "text-embedding-ada-002"):
        import openai
        self._model_id = model
        self._client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    @property
    def model_name(self): return self._model_id
    @property
    def provider_name(self): return "openai"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        r = self._client.embeddings.create(input=texts, model=self._model_id)
        return [item.embedding for item in r.data]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]


# ── Ollama ────────────────────────────────────────────

class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model: str = "nomic-embed-text"):
        import requests
        self._model_id = model
        self._base = settings.OLLAMA_BASE_URL
        self._req = requests

    @property
    def model_name(self): return self._model_id
    @property
    def provider_name(self): return "ollama"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        r = self._req.post(
            f"{self._base}/api/embeddings",
            json={"model": self._model_id, "prompt": query},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["embedding"]


# ── Factory ───────────────────────────────────────────

class EmbedderFactory:
    _registry = {
        "huggingface": HuggingFaceEmbedder,
        "openai": OpenAIEmbedder,
        "ollama": OllamaEmbedder,
    }

    @classmethod
    def create(cls, provider: str = settings.EMBEDDING_PROVIDER) -> BaseEmbedder:
        provider = provider.lower()
        if provider not in cls._registry:
            raise ValueError(f"Unknown embedding provider '{provider}'. Available: {list(cls._registry)}")
        return cls._registry[provider]()

    @classmethod
    def available(cls) -> List[str]:
        return list(cls._registry.keys())
