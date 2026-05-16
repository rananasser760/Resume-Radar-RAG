# ResumeRadar — AI-Powered Resume Screening System

> **RAG-based candidate intelligence** · Arabic + English · Groq · OpenRouter · Ollama · Docker-ready

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**ResumeRadar** is a production-grade Retrieval-Augmented Generation (RAG) system for intelligent resume screening. It accepts natural language queries in **English or Arabic**, semantically retrieves the most relevant candidate profiles from a ChromaDB vector store, and generates context-grounded answers through a fully configurable LLM backend.

The system is designed around four engineering principles:

- **Real-world PDF robustness** — custom PyMuPDF parser handles multi-column layouts, Arabic RTL text, and scanned fonts.
- **Cross-lingual retrieval** — a multilingual embedding model places Arabic and English text in the same vector space, so an Arabic query can match English résumés and vice versa.
- **Zero-downtime LLM switching** — a Factory Design Pattern lets you hot-swap between Ollama, OpenRouter, and Groq at runtime without restarting the server.
- **One-command deployment** — fully containerized with `docker-compose up --build`.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [LLM Providers](#llm-providers)
- [Embedding Model](#embedding-model)
- [Arabic Language Support](#arabic-language-support)
- [Evaluation & Known Edge Cases](#evaluation--known-edge-cases)
- [Running Tests](#running-tests)
- [Tech Stack](#tech-stack)

---

## Features

| Feature | Details |
|---|---|
| **RAG Pipeline** | Parse → Chunk → Embed → Store → Retrieve → Generate |
| **Multilingual** | Arabic (RTL) and English résumés in the same index |
| **LLM Factory** | Hot-swap Groq / OpenRouter / Ollama with a single API call |
| **Multi-Query Expansion** | Automatically expands Arabic queries to improve cross-lingual recall |
| **Metadata Baking** | Candidate name and language injected into every chunk for accurate LLM reasoning |
| **Interactive UI** | Built-in web interface served at `/` |
| **Swagger Docs** | Auto-generated at `/docs` |
| **Docker Compose** | Full stack (API + optional Ollama) with persistent ChromaDB volume |
| **Evaluation Suite** | 7 edge-case test scenarios with JSON report output |

---

## Architecture

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
│  │                     RAG Pipeline                        │    │
│  │                                                         │    │
│  │  raw_data/ ──► PDFParser ──► ChunkingService            │    │
│  │                 (fitz)        (500 / 50 chars)          │    │
│  │                    │                                    │    │
│  │             EmbedderFactory                             │    │
│  │      paraphrase-multilingual-MiniLM-L12-v2              │    │
│  │                    │                                    │    │
│  │           VectorStore (ChromaDB) ◄── query              │    │
│  │                    │                                    │    │
│  │           LLMFactory ──── Groq / OpenRouter / Ollama    │    │
│  │          (hot-swappable)                                │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

The codebase follows an **MVC separation** pattern:

| Layer | File | Responsibility |
|---|---|---|
| Model | `app/models/schemas.py` | Pydantic request/response validation |
| Controller | `app/api/routes.py` | FastAPI route handlers, HTTP logic |
| Service | `app/services/` | Business logic — parser, chunker, embedder, LLM, pipeline |
| Config | `app/core/config.py` | Environment-driven settings via `pydantic-settings` |

---

## Project Structure

```
ResumeRadarRAG/
├── app/
│   ├── main.py                  ← FastAPI entry point + lifespan hooks
│   ├── api/
│   │   └── routes.py            ← All API endpoints (Controller layer)
│   ├── core/
│   │   └── config.py            ← Env-driven settings
│   ├── models/
│   │   └── schemas.py           ← Pydantic request/response models
│   ├── services/
│   │   ├── parser.py            ← PyMuPDF parser with Arabic RTL fix
│   │   ├── chunker.py           ← RecursiveCharacterTextSplitter (500/50)
│   │   ├── embedder.py          ← EmbedderFactory (HuggingFace / OpenAI / Ollama)
│   │   ├── llm.py               ← LLMFactory (Ollama / OpenRouter / Groq)
│   │   ├── vector_store.py      ← ChromaDB wrapper
│   │   └── rag_pipeline.py      ← Full pipeline orchestrator
│   └── static/
│       └── index.html           ← Built-in web UI
├── raw_data/                    ← 📁 Place your PDF résumés here
├── chroma_db/                   ← Auto-created, persisted via Docker volume
├── tests/
│   ├── evaluation.py            ← Edge-case evaluation suite (7 test cases)
│   └── switch_llm_demo.py       ← Live provider-switching demo script
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env                         ← Your environment config (copy from .env.example)
└── TECHNICAL_REPORT.md          ← Full technical documentation
```

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 24.0 and Docker Compose ≥ 2.0
- At least one API key: [Groq](https://console.groq.com/keys) (free) or [OpenRouter](https://openrouter.ai/keys) (free tier available)
- 8 GB RAM only required if running Ollama locally — Groq/OpenRouter have no local RAM overhead

### 1. Configure your environment

```bash
cp .env .env.backup      # keep a backup
# Open .env and add your keys:
#   GROQ_API_KEY=gsk_...
#   OPENROUTER_API_KEY=sk-or-...
```

### 2. Add your résumé PDFs

```
./raw_data/
├── alice_johnson.pdf
├── ahmed_hassan.pdf
└── ...   (any number of PDFs — English or Arabic)
```

### 3. Start the service

```bash
docker-compose up --build
```

> First run downloads the embedding model (~500 MB) and, if using Ollama, the Mistral model (~4 GB).

### 4. Ingest résumés

```bash
curl -X POST http://localhost:8000/api/v1/ingest
```

### 5. Query in English

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who has Python and deep learning experience?", "top_k": 5}'
```

### 6. Query in Arabic

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "من لديه خبرة في تطوير تطبيقات الذكاء الاصطناعي؟"}'
```

### 7. Open interactive docs

```
http://localhost:8000/docs
```

---

## Configuration

All settings are controlled via the `.env` file. The full set of available options:


> **Groq/OpenRouter only (no GPU needed):** Comment out the `ollama` and `ollama-init` services in `docker-compose.yml` and set `LLM_PROVIDER=groq` or `LLM_PROVIDER=openrouter` in `.env`.

---

## LLM Providers

| Provider | Speed | Cost | Best For |
|---|---|---|---|
| **Groq** | ⚡ Ultra-fast | Free tier | Development, demos, fast iteration |
| **OpenRouter** | 🌐 Fast | Pay-per-token | Production, access to 100+ models |
| **Ollama** | 🦙 Moderate | Free (local) | Air-gapped environments, data privacy |

Switch at runtime — no restart required:

```bash
# Switch to OpenRouter
curl -X POST http://localhost:8000/api/v1/switch-llm \
  -H "Content-Type: application/json" \
  -d '{"provider": "openrouter", "model": "meta-llama/llama-3-8b-instruct"}'

# Switch to Groq
curl -X POST http://localhost:8000/api/v1/switch-llm \
  -H "Content-Type: application/json" \
  -d '{"provider": "groq", "model": "mixtral-8x7b-32768"}'

# Switch to local Ollama
curl -X POST http://localhost:8000/api/v1/switch-llm \
  -H "Content-Type: application/json" \
  -d '{"provider": "ollama", "model": "llama3"}'
```

**Available models per provider:**

| Provider | Example Models |
|---|---|
| `ollama` | `mistral`, `llama3`, `gemma`, `phi3`, `codellama` |
| `openrouter` | `mistralai/mistral-7b-instruct`, `meta-llama/llama-3-8b-instruct`, `google/gemma-7b-it:free`, `anthropic/claude-3-haiku` |
| `groq` | `llama3-8b-8192`, `llama3-70b-8192`, `mixtral-8x7b-32768`, `gemma-7b-it`, `llama-3.1-8b-instant` |

---

## Embedding Model

**Model:** `paraphrase-multilingual-MiniLM-L12-v2`

| Criterion | Justification |
|---|---|
| **Multilingual** | Trained on 50+ languages; Arabic and English share the same latent vector space — an Arabic query correctly retrieves English-language résumé chunks |
| **Lightweight** | 118M parameters — runs fully on CPU inside Docker, no GPU required |
| **Semantic quality** | Paraphrase-trained, so it captures that "machine learning" ≈ "ML" ≈ "AI models" |
| **Zero cost** | Local model; no API billing per embedding call |

---


## Arabic Language Support

ResumeRadar includes a complete Arabic NLP preprocessing pipeline:

1. **Detection** — `_arabic_ratio()` counts Arabic Unicode codepoints (U+0600–U+06FF) and flags a document as Arabic when the ratio exceeds 15%.
2. **Reshaping** — `arabic_reshaper.reshape()` converts isolated Arabic letter forms to their correct connected presentation forms (required for accurate text extraction from many Arabic-generated PDFs).
3. **BiDi algorithm** — `python-bidi`'s `get_display()` applies the Unicode Bidirectional Algorithm to reverse RTL text that PDF readers often extract in the wrong order.
4. **Chunking** — The Arabic comma `،` is added to the splitter's separator list.
5. **Cross-lingual retrieval** — `paraphrase-multilingual-MiniLM-L12-v2` places Arabic and English text in the same embedding space, enabling seamless cross-lingual matching.

**Example:**

```bash
# Arabic query retrieving English résumé chunks
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "من لديه خبرة في تطوير تطبيقات الويب؟"}'
```

Cross-lingual retrieval similarity score observed in testing: **0.81**.

---

## Evaluation & Known Edge Cases

Run the full evaluation suite (requires the API to be running):

```bash
python tests/evaluation.py
# → Runs 7 test cases (English + Arabic + edge cases)
# → Saves results to tests/eval_report.json
```



## Running Tests

```bash
# Run the edge-case evaluation suite
python tests/evaluation.py

# Run the live LLM provider switching demo
python tests/switch_llm_demo.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Web Framework** | FastAPI 0.111 + Uvicorn |
| **Data Validation** | Pydantic v2 + pydantic-settings |
| **PDF Parsing** | PyMuPDF (fitz) 1.24 |
| **Arabic NLP** | arabic-reshaper 3.0 + python-bidi 0.4 |
| **Text Chunking** | LangChain `RecursiveCharacterTextSplitter` |
| **Embeddings** | sentence-transformers 3.0 · `paraphrase-multilingual-MiniLM-L12-v2` |
| **Vector Database** | ChromaDB 0.5 (persisted volume) |
| **LLM Backends** | Groq SDK · OpenAI-compatible OpenRouter · Ollama REST API |
| **Containerization** | Docker + Docker Compose (multi-service, named volumes) |

---

## Repository

```
https://github.com/rananasser760/Resume-Radar-RAG
```

---

*Built with FastAPI · ChromaDB · sentence-transformers · Docker*

# 👩‍💻 Author

Developed by:
Rana Nasser
Ahmed Khaled
AbdElRahman Taher
Mohamed Saad
Mohamed Amir
Mohamed Khaled

Computer Science Student | AI & Software Development Enthusiast

---

# ⭐ Repository Support

If you found this project useful, consider giving it a ⭐ on GitHub.

