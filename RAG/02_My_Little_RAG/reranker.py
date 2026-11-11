"""Optional reranker add-on for the RAG/02 retrieval engine.

Hard add-on semantics (user constraint): when the reranker is disabled OR
unreachable, retrieval results are returned unchanged — same list object,
same order, same scores, no ``rerank_score`` key. Callers cannot distinguish
a disabled reranker from the pre-add-on engine by looking at the output.

Sync httpx client on purpose: engine calls already run inside
``asyncio.to_thread`` at the MCP layer, so blocking here is fine.

Two wire shapes, selected by ``RERANK_MODE`` (default ``listwise`` — the
historical behavior, so pre-RERANK_MODE deployments keep working):

- ``listwise`` — llama.cpp decision-model endpoint via the llama-swap model
  passthrough ``{RERANK_BASE_URL}/upstream/{RERANK_MODEL}/v1/systemone``.
  llama-swap proxies any method+path under ``/upstream/<model>/`` straight
  to that model's llama-server, which serves llama.cpp's ``POST
  /v1/systemone`` decision-model endpoint. The passthrough works on any
  llama-swap version; the bare ``/v1/systemone`` route would need
  llama-swap >= v262. Wire format (llama.cpp PR #29818 / HF blog "New in
  llama.cpp: Decision Models"): one request = one ``state`` + a map of
  questions. ``score`` questions take 2-10 levels lowest-first and return
  an expected level as ``answers[<qid>]["score"]`` (float, can fall between
  two levels). Reranking shape (benchmark-best pattern): listwise — state
  carries the query plus all numbered passages, one score question per
  passage, chunks ordered by expected level.

- ``rerank`` — llama.cpp encoder endpoint
  ``{RERANK_BASE_URL}/upstream/{RERANK_MODEL}/v1/rerank`` (llama-swap
  passthrough, same composition), body ``{"model", "query", "documents"[]}``.
  The response ``results[]`` entries carry ``index`` (position in the
  request's ``documents``) and ``relevance_score`` (float, higher = more
  relevant); entries arrive in relevance order, so they must be mapped back
  through ``index``. This is the cross-encoder shape served by reranking
  models (production: qwen3-rerank-06b).

Both modes share the same env gates, the same fail-open fallback, and the
same output contract: retrieval ``score`` untouched, ``rerank_score`` added,
``rank`` reassigned.
"""

import os

import httpx


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


# Configuration — os.getenv defaults, same pattern as retrieval_engine.py.
# OPTIONAL tuning knobs: NEVER add these to validate_required_env(); a
# missing or malformed value must degrade to "reranker off", never break
# service startup.
RERANK_ENABLED = _env_flag("RERANK_ENABLED")
RERANK_BASE_URL = (os.getenv("RERANK_BASE_URL") or "").rstrip("/")
RERANK_MODEL = os.getenv("RERANK_MODEL") or ""
RERANK_TIMEOUT_S = _env_float("RERANK_TIMEOUT_S", 3.0)
# Wire shape: "rerank" = encoder /v1/rerank endpoint; anything else
# (unset, unknown value) = the historical listwise /v1/systemone shape.
RERANK_MODE = (os.getenv("RERANK_MODE") or "listwise").strip().lower()

# Prompt shape verbatim from the public decision-model rerank benchmarks:
# 4-level relevance rubric, lowest first; order passages by expected level.
_SCORE_INSTRUCTIONS = (
    "How well does the passage supply the information needed "
    "to answer or verify the query?"
)
_SCORE_CRITERIA = [
    "off-topic",
    "related but does not supply",
    "partly supplies",
    "fully supplies",
]


def _build_listwise_payload(query: str, results: list) -> dict:
    """Listwise systemone payload: one state = query + all chunks, one score
    question per chunk (answered independently in a single batch)."""
    state = {
        "query": query,
        "passages": [
            {"id": i + 1, "text": r.get("content", "")}
            for i, r in enumerate(results)
        ],
    }
    questions = {
        str(i + 1): {
            "type": "score",
            "instructions": f"Passage {i + 1}: {_SCORE_INSTRUCTIONS}",
            "criteria": list(_SCORE_CRITERIA),
        }
        for i in range(len(results))
    }
    payload = {"state": state, "questions": questions}
    if RERANK_MODEL:
        payload["model"] = RERANK_MODEL
    return payload


def _parse_listwise_scores(data: dict, count: int) -> list:
    """One float per question id (1..count) from a systemone response.

    Raises KeyError/TypeError/ValueError on any shape mismatch — the caller
    treats that as a reranker failure (add-on semantics: fall back).
    """
    answers = data["answers"]
    return [float(answers[str(i + 1)]["score"]) for i in range(count)]


def _build_rerank_payload(query: str, results: list) -> dict:
    """Encoder /v1/rerank payload: flat query + document texts in input
    order; scoring is per (query, document) pair."""
    return {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [r.get("content", "") for r in results],
    }


def _parse_rerank_scores(data: dict, count: int) -> list:
    """One float per document position (0..count-1) from a /v1/rerank
    response.

    ``results[]`` arrives in relevance order, NOT input order — map back
    through ``index``. Raises KeyError/TypeError/ValueError on any shape
    mismatch (missing index, missing/garbage score, wrong count) — the
    caller treats that as a reranker failure (add-on semantics: fall back).
    """
    scores = {}
    for entry in data["results"]:
        scores[int(entry["index"])] = float(entry["relevance_score"])
    return [scores[i] for i in range(count)]


def maybe_rerank(query: str, results: list) -> list:
    """Rerank ``results`` by model relevance, or return them as-is.

    Gate: disabled, unconfigured, or fewer than 2 results -> the SAME list,
    untouched (no HTTP, no copies, no rerank_score key).
    """
    if (
        not RERANK_ENABLED
        or not RERANK_BASE_URL
        or not RERANK_MODEL
        or len(results) < 2
    ):
        return results
    if RERANK_MODE == "rerank":
        url = f"{RERANK_BASE_URL}/upstream/{RERANK_MODEL}/v1/rerank"
        payload = _build_rerank_payload(query, results)
        parse = _parse_rerank_scores
    else:
        url = f"{RERANK_BASE_URL}/upstream/{RERANK_MODEL}/v1/systemone"
        payload = _build_listwise_payload(query, results)
        parse = _parse_listwise_scores
    try:
        with httpx.Client(
            timeout=httpx.Timeout(RERANK_TIMEOUT_S, connect=1.0)
        ) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            scores = parse(resp.json(), len(results))
    except Exception as e:  # add-on semantics: ANY failure -> unranked path
        print(f"[RAG/02 reranker] disabled-fallback: {e}")
        return results  # original list object, unmodified, no rerank_score key
    order = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)
    out = [dict(results[i], rerank_score=scores[i]) for i in order]
    for rank, r in enumerate(out, 1):
        r["rank"] = rank  # reassign rank; keep original retrieval 'score'
    return out
