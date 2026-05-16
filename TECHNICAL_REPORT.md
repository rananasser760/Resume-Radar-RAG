# ResumeRadar — Technical Report

**AI-Powered Resume Screening System | RAG Architecture**
_Full Bonus Implementation_

---

## Executive Summary

ResumeRadar is a production-grade, fully containerized Retrieval-Augmented Generation (RAG) system for intelligent resume screening. Given a natural language query (English **or Arabic**), the system semantically retrieves the most relevant candidate profiles from a ChromaDB vector store and generates a context-grounded answer using a configurable LLM backend.

**Key engineering decisions:**

- Custom PyMuPDF parser handles real-world messy PDFs (multi-column, Arabic RTL, scanned fonts)
- Multilingual embedding model places Arabic and English text in the **same** vector space
- Factory Design Pattern enables zero-downtime LLM switching between Ollama, OpenRouter, and Groq
- Full Docker containerization — one `docker-compose up --build` command

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Compose                           │
│                                                                 │
│  ┌─────────────────────┐          ┌──────────────────────────┐  │
│  │  ResumeRadar FastAPI│◄────────►│  Ollama (local LLM)      │  │
│  │  :8000              │          │  :11434 / mistral        │  │
│  └──────────┬──────────┘          └──────────────────────────┘  │
│             │                                                   │
│  ┌──────────▼──────────────────────────────────────────────┐    │
│  │                    RAG Pipeline                         │    │
│  │                                                         │    │
│  │  raw_data/      →  PDFParser   →  ChunkingService       │    │
│  │  (13 PDFs)          (fitz)        (500/50 chars)        │    │
│  │                         ↓                               │    │
│  │                   EmbedderFactory                       │    │
│  │              paraphrase-multilingual-MiniLM-L12-v2      │    │
│  │                         ↓                               │    │
│  │                   VectorStore (ChromaDB)  ←── query     │    │
│  │                         ↓                               │    │
│  │                   LLMFactory  ──────── Groq / OpenRouter│    │
│  │               (hot-swappable)           / Ollama        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### MVC Separation

| Layer      | File                    | Responsibility                                            |
| ---------- | ----------------------- | --------------------------------------------------------- |
| Model      | `app/models/schemas.py` | Pydantic request/response validation                      |
| Controller | `app/api/routes.py`     | FastAPI route handlers, HTTP logic                        |
| Service    | `app/services/`         | Business logic — parser, chunker, embedder, LLM, pipeline |
| Config     | `app/core/config.py`    | Env-driven settings via pydantic-settings                 |

---

## API Documentation

### Base URL

```
http://localhost:8000/api/v1
```

Interactive docs: **http://localhost:8000/docs**

---

### `POST /ingest`

Triggers the full ingestion pipeline on `./raw_data/`.

**Request:**

```json
{
  "clear_before_ingest": false
}
```

**Response:**

```json
{
  "status": "success",
  "files_processed": 30,
  "chunks_stored": 387,
  "message": "Ingested 30 resumes → 387 chunks stored.",
  "file_details": [
    {
      "filename": "ahmed_cv.pdf",
      "pages": 2,
      "language": "arabic",
      "arabic_ratio": 0.72,
      "chunks": 14
    }
  ]
}
```

---

### `POST /query`

Full RAG pipeline — English or Arabic queries.

**Request:**

```json
{
  "query": "Who has 3+ years of Python and published ML research?",
  "top_k": 5,
  "llm_provider": "groq",
  "llm_model": "llama-3.1-8b-instant"
}
```

**Arabic Request:**

```json
{
  "query": "من لديه خبرة في تطوير تطبيقات الذكاء الاصطناعي؟",
  "top_k": 5
}
```

**Response:**

```json
{
  "query": "Who has 3+ years of Python and published ML research?",
  "answer": "Based on the provided resumes, Ahmed Hassan has 4 years of Python experience and co-authored two NLP papers at ACL 2023. Sara Khalil also lists Python as a primary skill with 3 years at a fintech startup.",
  "retrieved_chunks": [
    {
      "content": "Ahmed Hassan | Senior ML Engineer...",
      "source": "ahmed_hassan.pdf",
      "chunk_id": "ahmed_hassan_c0002",
      "score": 0.923,
      "metadata": { "language": "english", "pages": 2, "chunk_index": 2 }
    }
  ],
  "llm_provider": "groq",
  "llm_model": "llama3-8b-8192",
  "chunks_used": 5
}
```

---

### `POST /switch-llm`

Hot-swap the active LLM **without restarting the server**.

**Request:**

```json
{ "provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct" }
```

**Response:**

```json
{
  "status": "success",
  "provider": "openrouter",
  "model": "mistralai/mistral-7b-instruct",
  "message": "Switched to meta-llama/llama-3.3-70b-instruct"
}
```

**Available providers + models:**

| Provider     | Example Models                                                                               |
| ------------ | -------------------------------------------------------------------------------------------- |
| `ollama`     | `mistral`, `llama3`, `gemma`, `phi3`                                                         |
| `openrouter` | `mistralai/mistral-7b-instruct`, `meta-llama/llama-3-8b-instruct`, `google/gemma-7b-it:free` |
| `groq`       | `llama3-8b-8192`, `llama3-70b-8192`, `mixtral-8x7b-32768`                                    |

---

### `GET /stats`

```json
{
  "collection_name": "resumes",
  "total_chunks": 387,
  "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
  "embedding_provider": "huggingface",
  "default_llm_provider": "groq",
  "default_llm_model": "llama3-8b-8192",
  "available_llm_providers": ["ollama", "openrouter", "groq"],
  "available_embedding_providers": ["huggingface", "openai", "ollama"]
}
```

### `GET /providers` — list all providers and active config

### `DELETE /clear` — wipe ChromaDB collection

---

## Embedding Model Justification

**Model:** `paraphrase-multilingual-MiniLM-L12-v2`

| Criterion              |             Justification J                                                                                                                                       |
| --------------------   | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Multilingual**     | Trained on 50+ languages; Arabic and English share the same latent space → a Groq query in Arabic retrieves English-language resume chunks |
| **Size**             | 118M parameters — runs fully on CPU inside Docker, no GPU required                                                                         |
| **Semantic quality** | Trained on paraphrase pairs → captures "machine learning" ≈ "ML" ≈ "AI models"                                                             |
| **Zero cost**        | Local model, no API billing on every embed call                                                                                            |

---

## Chunking Strategy Justification

**Parameters: 500 chars / 50 overlap**

```
"Senior Data Engineer @ Vodafone Egypt (2020–2023)
 Built real-time Kafka pipelines processing 500K events/day.
 Skills: Python, Spark, Airflow, AWS S3."
 ← fits in ~300 chars — stays in ONE chunk ✅
```

| Parameter        | Value                            | Reasoning  R                                                                       |
| ---------------- | -------------------------------- | --------------------------------------------------------------------------------- |
| Chunk size       | 500 chars                        | Covers ~2–3 sentences — one full job entry or skills section                      |
| Overlap          | 50 chars                         | Prevents boundary splits mid-sentence (e.g., job title separated from date range) |
| Splitter         | `RecursiveCharacterTextSplitter` | Hierarchical: `\n\n → \n → . → ، → " "` — respects paragraph structure            |
| Arabic separator | `،`                              | Handles Arabic comma-separated skill lists correctly                              |

---

## Phase 4: Evaluation & Error Analysis

Run full evaluation: `python tests/evaluation.py`

### Edge Case 1 — Semantic keyword overlap (EDGE-01)

**Query:** _"Who has an arabic CV?"_

**Failure observed:** LLM listed people who only wrote the word arabic in their CV instead of people who have an arabic cv.

**Root cause:** The LLM suffers from semantic keyword overlap and lacks structural document awareness. When the retriever pulls a chunk containing the phrase "Languages: Arabic (Native)", the LLM assumes this satisfies the query. It cannot inherently distinguish between a document containing the word "Arabic" and a document written in the Arabic language.

**Fix:** Implement Metadata Baking combined with Strict System Prompting.

---

### Edge Case 2 — Multi-page Fragmentation (EDGE-02)

**Query:** _"List candidates who graduated from Cairo University."_

**Failure observed:** Missed 3 candidates whose university was on page 2 while their name was on page 1. Chunking split these into separate vectors with no linking context.

**Root cause:** RecursiveCharacterTextSplitter has no document-level awareness — candidate name and their degree end up in different chunks.

**Fix:** Extract candidate name as a metadata field during parsing and inject it into every chunk's text prefix: `"[Candidate: Ahmed Hassan] Senior ML Engineer..."` — this ensures every chunk is self-contained and retrievable.

---

### Edge Case 3 — Synonym / Terminology Gap (EDGE-03)

**Query:** _"Find candidates with cloud infrastructure experience."_

**Failure observed:** Retrieved only candidates who explicitly wrote "cloud" — missed those who listed "AWS", "GCP", "Azure", "EC2", "S3" without the word "cloud."

**Root cause:** The embedding model captures semantic proximity, but domain-specific abbreviations (AWS ≠ cloud in the model's learned representation) create retrieval gaps.

**Fix:** Implement hybrid retrieval — combine dense vector search (semantic) with sparse BM25 keyword search. BM25 would directly match "AWS" while the dense model handles semantic variants.

---

## Bonus Features

### ✅ Bonus 1 — LLM Factory Pattern (+5%)

Implemented in `app/services/llm.py`:

```python
factory = LLMFactory()           # loads env default (e.g., groq)
factory.switch("openrouter")     # zero-restart provider swap
factory.switch("ollama", "llama3")
```

- `LLMFactory._registry` maps string keys to provider classes
- `switch()` instantiates the new class, validates the API key, and replaces `_current`
- Exposed via `POST /api/v1/switch-llm` for runtime control
- Per-request override also available via `llm_provider` field in `/query`

### ✅ Bonus 2 — Arabic Language Support (+10%)

**Arabic NLP pipeline:**

1. **Detection:** `_arabic_ratio()` counts Arabic Unicode codepoints (U+0600–U+06FF) — triggers if >15%
2. **Reshaping:** `arabic_reshaper.reshape()` converts isolated Arabic letter forms to their connected presentation forms (required for correct text extraction from many Arabic PDFs)
3. **BiDi algorithm:** `get_display()` from `python-bidi` applies the Unicode Bidirectional Algorithm — correctly reverses RTL text that PDF readers extract in wrong order
4. **Chunking:** Added `،` (Arabic comma) to separator list
5. **Cross-lingual retrieval:** `paraphrase-multilingual-MiniLM-L12-v2` embeds Arabic and English into the same vector space

**Arabic test case result:**

Query: `"من لديه خبرة في تطوير تطبيقات الذكاء الاصطناعي؟"`

Retrieved chunks correctly matched English-language ML resumes — demonstrating cross-lingual semantic retrieval working as designed. Score: 0.81.

---

## Docker Deployment

### Prerequisites

- Docker ≥ 24.0 and Docker Compose ≥ 2.0
- 8 GB RAM (for Mistral) — or skip Ollama and use Groq/OpenRouter only

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/rananasser760/Resume-Radar-RAG.git
cd Resume-Radar-RAG

# 2. Configure environment
cp .env.example .env
# Edit .env → add GROQ_API_KEY and/or OPENROUTER_API_KEY

# 3. Drop your 30 PDF resumes into:
#    ./raw_data/

# 4. Start all services
docker-compose up --build
# First run downloads Mistral (~4 GB) and embedding model (~500 MB)

# 5. Ingest resumes
curl -X POST http://localhost:8000/api/v1/ingest

# 6. Query (English)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who has Python and ML experience?", "top_k": 5}'

# 7. Query (Arabic)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "من لديه خبرة في تعلم الآلة؟", "top_k": 5}'

# 8. Switch to Groq at runtime
curl -X POST http://localhost:8000/api/v1/switch-llm \
  -H "Content-Type: application/json" \
  -d '{"provider": "groq", "model": "mixtral-8x7b-32768"}'

# 9. Interactive API docs
open http://localhost:8000/docs
```

### To use Groq/OpenRouter only (no GPU/RAM needed for Ollama)

Comment out the `ollama` and `ollama-init` services in `docker-compose.yml`
and set `LLM_PROVIDER=groq` in `.env`.
