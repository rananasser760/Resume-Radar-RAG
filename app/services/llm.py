"""
LLM Factory Pattern — Bonus Phase 1
Providers: ollama | openrouter | groq
- Runtime switching via /api/v1/switch-llm endpoint (no restart needed)
- Each provider wraps a unified BaseLLM interface
- RAG prompt template is shared across all providers
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from app.core.config import settings


# ── Shared RAG Prompt ─────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a technical recruitment assistant for ResumeRadar.
Your goal is to analyze CVs based on the provided context.
Be specific, cite candidate names, and avoid hallucinating any information not in the context.
If the context does not contain enough information, clearly say so.

CRITICAL INSTRUCTIONS FOR LANGUAGE RECOGNITION:
1. A CV is only "written in Arabic" if the CONTENT itself uses Arabic script (e.g., مريم محمد).
2. If a CV is written in English but lists "Arabic: Native" in a skills section, it is an ENGLISH CV.
3. Always check the 'Language' metadata tag provided in the context blocks.
4. If the content looks like scrambled characters or 'ᒚᓚᒠ', treat it as a parsing error.
"""

RAG_USER_TEMPLATE = """RESUME CONTEXT:
{context}

---
QUESTION: {query}

Answer based strictly on the context above. Include candidate names when relevant."""


def build_prompt(query: str, context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return RAG_USER_TEMPLATE.format(context=context, query=query)


# ── Abstract Base ─────────────────────────────────────

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, query: str, context_chunks: List[str]) -> str: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...


# ── Ollama (local) ────────────────────────────────────

class OllamaLLM(BaseLLM):
    """Local Ollama server — e.g., mistral, llama3, gemma."""

    DEFAULT_MODEL = "mistral"

    def __init__(self, model: Optional[str] = None):
        import requests
        self._model = model or settings.OLLAMA_MODEL
        self._base = settings.OLLAMA_BASE_URL
        self._req = requests
        print(f"[LLM] 🦙 Ollama | model: {self._model}")

    @property
    def provider_name(self): return "ollama"
    @property
    def model_name(self): return self._model

    def generate(self, query: str, context_chunks: List[str]) -> str:
        prompt = build_prompt(query, context_chunks)
        full = f"{RAG_SYSTEM_PROMPT}\n\n{prompt}"
        try:
            r = self._req.post(
                f"{self._base}/api/generate",
                json={"model": self._model, "prompt": full, "stream": False},
                timeout=180,
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}") from e


# ── OpenRouter ────────────────────────────────────────

class OpenRouterLLM(BaseLLM):
    """
    OpenRouter — routes to hundreds of models (Mistral, Claude, GPT-4, etc.)
    via a unified OpenAI-compatible API.
    Docs: https://openrouter.ai/docs
    """

    DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"

    def __init__(self, model: Optional[str] = None):
        import openai
        self._model = model or settings.OPENROUTER_MODEL
        api_key = settings.OPENROUTER_API_KEY
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in .env")
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=settings.OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://ResumeRadar.ai",
                "X-Title": "ResumeRadar Resume Screener",
            },
        )
        print(f"[LLM] 🌐 OpenRouter | model: {self._model}")

    @property
    def provider_name(self): return "openrouter"
    @property
    def model_name(self): return self._model

    def generate(self, query: str, context_chunks: List[str]) -> str:
        prompt = build_prompt(query, context_chunks)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenRouter error: {e}") from e


# ── Groq ──────────────────────────────────────────────

class GroqLLM(BaseLLM):
    """
    Groq Cloud — ultra-fast inference via LPU chips.
    Available models: llama-3.1-8b-instant
    Docs: https://console.groq.com/docs
    """

    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(self, model: Optional[str] = None):
        from groq import Groq
        self._model = model or settings.GROQ_MODEL
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in .env")
        self._client = Groq(api_key=api_key)
        print(f"[LLM] ⚡ Groq | model: {self._model}")

    @property
    def provider_name(self): return "groq"
    @property
    def model_name(self): return self._model

    def generate(self, query: str, context_chunks: List[str]) -> str:
        prompt = build_prompt(query, context_chunks)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Groq error: {e}") from e


# ── Factory ───────────────────────────────────────────

class LLMFactory:
    """
    Factory Design Pattern for LLM providers.

    Runtime switching example:
        factory = LLMFactory()
        factory.switch("groq", model="mixtral-8x7b-32768")
        answer = factory.current.generate(query, chunks)
    """

    _registry = {
        "ollama": OllamaLLM,
        "openrouter": OpenRouterLLM,
        "groq": GroqLLM,
    }

    # Sensible defaults per provider
    _defaults = {
        "ollama": OllamaLLM.DEFAULT_MODEL,
        "openrouter": OpenRouterLLM.DEFAULT_MODEL,
        "groq": GroqLLM.DEFAULT_MODEL,
    }

    def __init__(self):
        self._current: Optional[BaseLLM] = None
        # Initialize with env-configured provider
        self.switch(settings.LLM_PROVIDER)

    @property
    def current(self) -> BaseLLM:
        return self._current

    def switch(self, provider: str, model: Optional[str] = None) -> BaseLLM:
        """Hot-swap the active LLM provider at runtime."""
        provider = provider.lower()
        if provider not in self._registry:
            raise ValueError(
                f"Unknown provider '{provider}'. Available: {list(self._registry)}"
            )
        resolved_model = model or self._defaults.get(provider)
        self._current = self._registry[provider](resolved_model)
        print(f"[LLMFactory] 🔄 Switched → {provider}/{self._current.model_name}")
        return self._current

    @classmethod
    def available(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def default_models(cls) -> dict:
        return {
            "ollama": OllamaLLM.DEFAULT_MODEL,
            "openrouter": OpenRouterLLM.DEFAULT_MODEL,
            "groq": GroqLLM.DEFAULT_MODEL,
        }
