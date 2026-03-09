# My Little RAG Ingestion

RAG document ingestion tool for processing markdown/text files and storing them in Qdrant.

## Overview

This tool processes documents from the `./input` directory, chunks them, creates embeddings (both dense and sparse), and stores them in Qdrant. Each subfolder in `./input` creates a separate collection in Qdrant.

The container runs once and exits after processing is complete.

## Prerequisites

The main docker-compose stack must be running before using this tool, as it requires:
- **Qdrant** - For vector storage
- **Docker network `ollama-tools`** - For communication with Qdrant

## Setup

### Configuration

Ensure the `.env` file exists and is configured:

```bash
cd RAG/01_My_Little_RAG_Ingestion
# Copy from RAG folder or create symlink
cp ../.env .env
```

Configure the required environment variables in `.env`:
- `PROVIDER` - Embedding provider: `openai` or `ollama`
- `QDRANT_HOST` - Qdrant host address (default: `qdrant`)
- `QDRANT_PORT` - Qdrant port (default: `6333`)
- `EMBEDDING_MODEL_NAME` - Name of embedding model
- `VECTOR_SIZE` - Embedding dimension (default: `1024`)
- `OPENAI_API_KEY` - API key for OpenAI-compatible services (if using OpenAI provider)
- `OPENAI_BASE_URL` - Base URL for OpenAI-compatible API (if using OpenAI provider)
- `OLLAMA_BASE_URL` - Ollama service URL (if using Ollama provider)

### Prepare Input Files

Create the `./input` directory and add your `.md`, `.txt`, or `.html` files:

```bash
mkdir -p ./input
```

**Collection Organization:** Each subfolder within `./input` creates a separate Qdrant collection:

```
./input/
├── docs/                # Creates collection named "docs"
│   ├── file1.md
│   └── file2.md
├── laws/                # Creates collection named "laws"
│   └── law1.md
└── file3.md             # Uses default collection
```

## Usage

### Running the Ingestion

From the `RAG/01_My_Little_RAG_Ingestion` directory:

```bash
docker-compose run --rm my-rag-ingest-only
```

The container will:
1. Scan `./input` for new/changed files
2. Chunk documents into paragraphs
3. Generate embeddings
4. Store vectors in Qdrant
5. Exit when complete

### Force Reindex

To reprocess all documents (ignore cached state):

```bash
docker-compose run -e FORCE_REINDEX=true --rm my-rag-ingest-only
```

## Usage Example: Ingest Open WebUI Documentation

Clone the Open WebUI documentation repository and ingest it:

```bash
# Navigate to the ingestion tool directory
cd RAG/01_My_Little_RAG_Ingestion

# Create input directory
mkdir -p ./input

# Clone the Open WebUI docs
git clone https://github.com/open-webui/docs.git ./input/openwebui-docs

# Run ingestion (this will create a collection named "openwebui-docs")
docker-compose run --rm my-rag-ingest-only
```

## Chunking Behavior

Documents are intelligently chunked based on their structure:

**Normal paragraphs** (≤ 600 tokens):
- Multiple paragraphs are grouped together in a buffer
- Chunks are emitted when the buffer reaches 200 tokens (MIN_CHUNK_SIZE)
- Paragraphs within a chunk are joined with double newlines, preserving the original paragraph structure

**Oversized paragraphs** (> 600 tokens):
- Split at sentence boundaries using `split_sentences_respecting_bounds()`
- Sentences are grouped together, respecting the 600 token limit
- Sentences within a chunk are joined with spaces
- Legal abbreviations (Abs., Nr., Art., u.a., etc.) are preserved during sentence splitting to prevent incorrect breaks

**Additional features:**
- YAML front matter is automatically removed before chunking
- Single extremely long sentences (> 600 tokens) are split word-by-word as a fallback
- MIN_CHUNK_SIZE (default: 200), MAX_CHUNK_SIZE (default: 600), and CHUNK_OVERLAP can be configured via environment variables

## Incremental Processing

The tool tracks processed files and only reprocesses documents that have:
- Been modified
- Been moved to a different subfolder (new collection)
- Not been successfully embedded previously
