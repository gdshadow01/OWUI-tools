# My Little RAG (RAG/02) — Qdrant retrieval service

Hybrid retrieval service over Qdrant (dense + sparse search), exposed through
two transports backed by the same engine (`retrieval_engine.py`):

- **MCP** — `retrieval_api.py` (tools `search`, `search_by_file`,
  `text_search`, `text_search_by_file`, `list_collections`)
- **OpenAPI** — `openapi_routes.py` (endpoints `/api/search`,
  `/api/search_by_file`, `/api/text_search`, `/api/text_search_by_file`,
  `/api/collections`)

This README documents one behavior difference between the transports: which
response fields each one exposes for the hybrid search endpoints. Configuration
and deployment live in the compose files and `.env.example`; the reranker's
wire formats are documented in `reranker.py`.

## Hybrid search response fields and reranking

The engine applies optional reranking (`reranker.py`) to the hybrid endpoints
`search` and `search_by_file` only. When reranking runs successfully it
re-sorts the results by model relevance, adds a `rerank_score` to each result,
and reassigns `rank`. The retrieval `score` is never modified.

| Hybrid-search outcome | MCP `search` / `search_by_file` | OpenAPI `/api/search` / `/api/search_by_file` |
| --- | --- | --- |
| Reranking succeeds | Reordered results; `rerank_score` included | Same reranked ordering, but `SearchResult` omits `rerank_score` |
| Reranking skipped or fails open | Original retrieval results; no added `rerank_score` | Original retrieval results; no `rerank_score` |

The asymmetry is structural: the MCP tools return the engine's result
dictionaries unchanged, while the OpenAPI handlers validate every result
against the `SearchResult` response model (`openapi_routes.py`), which keeps
only `rank`, `score`, `content`, `file_name`, and `collection_name`. An
engine-added `rerank_score` is dropped there even when reranking succeeded.

### Field semantics

- **`score`** — the retrieval (hybrid search) score, not the reranker's
  relevance score. Reranking never changes it, so after a successful rerank it
  is no longer monotonic with the returned order. Do not re-sort results by
  `score`; to preserve the returned (reranked) ranking, keep the array order /
  use `rank`.
- **`rank` and array order** — reflect the returned ranking. Successful
  reranking reassigns `rank` in relevance order, so non-monotonic `score`
  values across positions are consistent with successful reranking. Treat that
  pattern as a hint, not a reliable standalone success detector: absence of
  reordering does not prove reranking was skipped.
- **`rerank_score`** — conditional, and visible over MCP only. It is present
  only when reranking actually ran and succeeded. It is absent when reranking
  is disabled or unconfigured (`RERANK_ENABLED` off, no reranker base URL or
  model), when fewer than two results are returned, or when the rerank request
  fails (unreachable endpoint, timeout, HTTP error, malformed response) —
  every handled rerank failure returns the retrieval results unchanged.

### What a missing `rerank_score` can and cannot tell you

Over MCP, a missing `rerank_score` means you got unreranked retrieval output;
by fail-open design the response does not say why. Over OpenAPI, a missing
`rerank_score` is not a signal at all: the response model drops the field even
when reranking succeeded, so OpenAPI responses alone cannot distinguish a
timeout, disabled reranking, or an inference failure from a successful rerank.

### Scope

This concerns the hybrid endpoints `search` / `search_by_file` only. The
keyword-only `text_search` / `text_search_by_file` endpoints are not reranked
on either transport and never carry `rerank_score`.

If an OpenAPI consumer needs explicit rerank scores or rerank status, that
requires a separately approved schema/observability change (for example an
optional response field). Do not overload `score` as a workaround — it is the
retrieval score, not a relevance score.
