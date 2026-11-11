import os
from typing import Optional, List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
)
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from reranker import RERANK_ENABLED, maybe_rerank

# Configuration
# Deliberately re-declared in RAG/01_My_Little_RAG_Ingestion/config.py (self-contained
# subtools, no shared config module). When changing a default here, change the mirror
# there and keep both aligned. The OPENAI_BASE_URL/OPENAI_RETRIEVAL_URL split is
# intentional divergence — see the OpenAI comment below.

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks")

# Provider
PROVIDER = os.getenv("PROVIDER", "openai").lower()  # "openai" or "ollama"

# Embedding
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "CPP_snowflake-embed-l-v2.0-GGUF")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 1024))  # Embedding dimension

# OpenAI
# Retrieval prefers OPENAI_RETRIEVAL_URL (a dedicated embedding host) and falls
# back to OPENAI_BASE_URL if unset. Reason for the split: BASE_URL may co-host
# the main LLM, and loading the embedding model there would UNLOAD it. A
# dedicated retrieval host avoids that. Ingestion (RAG/01) uses BASE_URL only,
# where batch unloading of the main LLM is acceptable.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_RETRIEVAL_URL") or os.getenv("OPENAI_BASE_URL")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

# BM25 configuration
BM25_LANGUAGE = os.getenv("BM25_LANGUAGE", "german")

# Reranker add-on (reranker.py) — optional candidate-set widening. When the
# reranker is enabled AND RERANK_CANDIDATES > top_k, the Qdrant RRF limit
# (and prefetch limits) are raised to it; the candidate list is cut back to
# top_k after scoring. 0/empty = in-place arm (rerank exactly the returned
# top_k). OPTIONAL knob: never in validate_required_env(); malformed values
# degrade to the in-place arm, they must not affect startup.
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


RERANK_CANDIDATES = _env_int("RERANK_CANDIDATES", 0)


def validate_required_env():
    required = ["QDRANT_HOST", "QDRANT_COLLECTION", "EMBEDDING_MODEL_NAME", "PROVIDER"]
    provider = os.environ.get("PROVIDER", "").lower()
    if provider == "openai":
        required.append("OPENAI_BASE_URL")
    elif provider == "ollama":
        required.append("OLLAMA_BASE_URL")
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        raise SystemExit(
            f"[RAG/02 retrieval] Missing required environment variables: {', '.join(missing)}\n"
            "Copy RAG/.env.example to the compose project .env and set the listed variables."
        )


validate_required_env()

def _build_bm25_query(text: str):
    options = {"language": BM25_LANGUAGE.lower()} if BM25_LANGUAGE and BM25_LANGUAGE.lower() != "none" else None
    return models.Document(
        text=text,
        model="qdrant/bm25",
        options=options,
    )

def get_embedding_function():
    """Get the appropriate embedding function based on the provider"""
    if PROVIDER == "ollama":
        return OllamaEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            base_url=OLLAMA_BASE_URL
        )
    elif PROVIDER == "openai":
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
    else:
        raise ValueError(f"Unsupported provider: {PROVIDER}")


class RetrievalEngine:
    """Class that encapsulates all retrieval functionality"""

    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
        self.embedding_func = get_embedding_function()

    def _rerank_fetch_limit(self, top_k: int) -> int:
        """Qdrant fetch width for hybrid queries: top_k, raised to
        RERANK_CANDIDATES when the reranker is enabled and asks for a wider
        candidate set (cut back to top_k after scoring). With the reranker
        disabled this is always top_k — the Qdrant call shape stays
        byte-identical to the pre-add-on engine."""
        if RERANK_ENABLED and RERANK_CANDIDATES > top_k:
            return RERANK_CANDIDATES
        return top_k

    def _maybe_rerank(self, query: str, results: List[dict]) -> List[dict]:
        """Optional rerank pass — no-op unless RERANK_ENABLED is truthy;
        on any reranker failure the original list is returned unchanged
        (see reranker.maybe_rerank)."""
        return maybe_rerank(query, results)

    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all available Qdrant collections

        Returns:
            List of collection information including name and point count
        """
        try:
            collections = self.client.get_collections()
            collection_list = []
            for collection in collections.collections:
                collection_name = collection.name
                # Now get detailed information about each collection
                try:
                    detailed_info = self.client.get_collection(collection_name)
                    points_count = getattr(detailed_info, 'points_count', 0)
                    vectors_count = getattr(detailed_info, 'vectors_count', 0)
                    indexed_vectors_count = getattr(detailed_info, 'indexed_vectors_count', 0)

                    # Get vector configuration to understand hybrid setup - try different ways to access config
                    try:
                        # Try to access configuration details based on Qdrant client structure
                        config = getattr(detailed_info, 'config', {})
                        if hasattr(config, 'params'):
                            params = getattr(config, 'params', {})
                            # Handle different possible formats for vectors and sparse vectors
                            if hasattr(params, 'vectors'):
                                vector_config = getattr(params, 'vectors', {})
                            else:
                                vector_config = params.get('vectors', params.get('vector', {}))

                            if hasattr(params, 'sparse_vectors'):
                                sparse_config = getattr(params, 'sparse_vectors', {})
                            else:
                                sparse_config = params.get('sparse_vectors', params.get('sparse', {}))
                        else:
                            # If config.params doesn't exist, check for direct access
                            vector_config = getattr(config, 'vectors', getattr(config, 'vector', {}))
                            sparse_config = getattr(config, 'sparse_vectors', getattr(config, 'sparse', {}))
                    except:
                        # Fallback in case of attribute errors
                        vector_config = {}
                        sparse_config = {}

                    # Note: In hybrid configurations, vectors_count might not reflect all stored vectors
                    # This is expected behavior when using both dense and sparse vectors
                    has_dense_vectors = bool(vector_config)
                    has_sparse_vectors = bool(sparse_config)

                except Exception as detail_error:
                    # If we can't get detailed info for a collection, log it but continue
                    print(f"Error getting details for collection {collection_name}: {str(detail_error)}")
                    points_count = 0
                    vectors_count = 0
                    indexed_vectors_count = 0
                    has_dense_vectors = False
                    has_sparse_vectors = False

                collection_info = {
                    "name": collection_name,
                    "point_count": points_count,
                    "vectors_count": vectors_count,
                    "indexed_vectors_count": indexed_vectors_count,
                    "has_dense_vectors": has_dense_vectors,
                    "has_sparse_vectors": has_sparse_vectors
                }
                collection_list.append(collection_info)
            return collection_list
        except Exception as e:
            print(f"Error listing collections: {str(e)}")
            return [{"error": str(e)}]

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        collection_name: Optional[str] = None
    ) -> List[dict]:
        """
        General hybrid search across all documents in the Qdrant collection.

        Args:
            query: The search query text
            top_k: Number of top results to return (default 10)
            min_score: Minimum similarity score for results (default 0.0)
            collection_name: Name of the collection to search in (defaults to main collection)

        Returns:
            List of search results with id, score, payload, and content
        """
        try:
            collection_to_search = collection_name if collection_name else QDRANT_COLLECTION

            dense_vector = self.embedding_func.embed_query(query)

            fetch_limit = self._rerank_fetch_limit(top_k)
            try:
                search_result = self.client.query_points(
                    collection_name=collection_to_search,
                    prefetch=[
                        models.Prefetch(
                            query=dense_vector,
                            using="dense",
                            limit=fetch_limit * 2,
                        ),
                        models.Prefetch(
                            query=_build_bm25_query(query),
                            using="sparse",
                            limit=fetch_limit * 3,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=fetch_limit,
                    with_payload=True
                ).points

            except Exception as e:
                print(f"Native hybrid search failed: {e}")
                search_result = self.client.query_points(
                    collection_name=collection_to_search,
                    query=dense_vector,
                    using="dense",
                    query_filter=None,
                    limit=top_k,
                    with_payload=True
                ).points

            results = [
                {
                    "rank": idx + 1,
                    "score": point.score,
                    "content": point.payload.get("text", "") if point.payload else "",
                    "file_name": point.payload.get("source", "") if point.payload else "",
                    "collection_name": collection_to_search
                }
                for idx, point in enumerate(search_result)
            ]

            if min_score > 0:
                results = [r for r in results if r["score"] >= min_score]

            results = self._maybe_rerank(query, results)
            if len(results) > top_k:
                # Widened candidate arm: cut back to top_k after scoring.
                results = results[:top_k]

            return results

        except Exception as e:
            print(f"Error during search: {str(e)}")
            return [{"error": str(e)}]

    def search_by_file(
        self,
        query: str,
        file_name: str,
        top_k: int = 10,
        min_score: float = 0.0,
        collection_name: Optional[str] = None
    ) -> List[dict]:
        """
        Hybrid search within a specific file in the Qdrant collection.

        Args:
            query: The search query text
            file_name: The specific file to search within
            top_k: Number of top results to return (default 10)
            min_score: Minimum similarity score for results (default 0.0)
            collection_name: Name of the collection to search in (defaults to main collection)

        Returns:
            List of search results with id, score, payload, and content from the specified file
        """
        try:
            collection_to_search = collection_name if collection_name else QDRANT_COLLECTION

            dense_vector = self.embedding_func.embed_query(query)

            file_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=file_name)
                    )
                ]
            )

            fetch_limit = self._rerank_fetch_limit(top_k)
            try:
                search_result = self.client.query_points(
                    collection_name=collection_to_search,
                    prefetch=[
                        models.Prefetch(
                            query=dense_vector,
                            using="dense",
                            limit=fetch_limit * 2,
                            filter=file_filter,
                        ),
                        models.Prefetch(
                            query=_build_bm25_query(query),
                            using="sparse",
                            limit=fetch_limit * 3,
                            filter=file_filter,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=fetch_limit,
                    with_payload=True
                ).points

            except Exception as e:
                print(f"Native hybrid search failed: {e}")
                search_result = self.client.query_points(
                    collection_name=collection_to_search,
                    query=dense_vector,
                    using="dense",
                    query_filter=file_filter,
                    limit=top_k,
                    with_payload=True
                ).points

            results = [
                {
                    "rank": idx + 1,
                    "score": point.score,
                    "content": point.payload.get("text", "") if point.payload else "",
                    "file_name": point.payload.get("source", "") if point.payload else "",
                    "collection_name": collection_to_search
                }
                for idx, point in enumerate(search_result)
            ]

            if min_score > 0:
                results = [r for r in results if r["score"] >= min_score]

            results = self._maybe_rerank(query, results)
            if len(results) > top_k:
                # Widened candidate arm: cut back to top_k after scoring.
                results = results[:top_k]

            return results

        except Exception as e:
            print(f"Error during search by file: {str(e)}")
            return [{"error": str(e)}]

    def text_search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        collection_name: Optional[str] = None
    ) -> List[dict]:
        """
        Text-only search using native BM25 sparse vectors across all documents in the Qdrant collection.

        Args:
            query: The search query text
            top_k: Number of top results to return (default 10)
            min_score: Minimum similarity score for results (default 0.0)
            collection_name: Name of the collection to search in (defaults to main collection)

        Returns:
            List of search results with id, score, payload, and content
        """
        try:
            collection_to_search = collection_name if collection_name else QDRANT_COLLECTION

            search_result = self.client.query_points(
                collection_name=collection_to_search,
                query=_build_bm25_query(query),
                using="sparse",
                limit=top_k,
                with_payload=True
            ).points

            results = [
                {
                    "rank": idx + 1,
                    "score": point.score,
                    "content": point.payload.get("text", "") if point.payload else "",
                    "file_name": point.payload.get("source", "") if point.payload else "",
                    "collection_name": collection_to_search
                }
                for idx, point in enumerate(search_result)
            ]

            if min_score > 0:
                results = [r for r in results if r["score"] >= min_score]

            return results

        except Exception as e:
            print(f"Error during text search: {str(e)}")
            return [{"error": str(e)}]

    def text_search_by_file(
        self,
        query: str,
        file_name: str,
        top_k: int = 10,
        min_score: float = 0.0,
        collection_name: Optional[str] = None
    ) -> List[dict]:
        """
        Text-only search using native BM25 sparse vectors within a specific file in the Qdrant collection.

        Args:
            query: The search query text
            file_name: The specific file to search within
            top_k: Number of top results to return (default 10)
            min_score: Minimum similarity score for results (default 0.0)
            collection_name: Name of the collection to search in (defaults to main collection)

        Returns:
            List of search results with id, score, payload, and content from the specified file
        """
        try:
            collection_to_search = collection_name if collection_name else QDRANT_COLLECTION

            file_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=file_name)
                    )
                ]
            )

            search_result = self.client.query_points(
                collection_name=collection_to_search,
                query=_build_bm25_query(query),
                using="sparse",
                query_filter=file_filter,
                limit=top_k,
                with_payload=True
            ).points

            results = [
                {
                    "rank": idx + 1,
                    "score": point.score,
                    "content": point.payload.get("text", "") if point.payload else "",
                    "file_name": point.payload.get("source", "") if point.payload else "",
                    "collection_name": collection_to_search
                }
                for idx, point in enumerate(search_result)
            ]

            if min_score > 0:
                results = [r for r in results if r["score"] >= min_score]

            return results

        except Exception as e:
            print(f"Error during text search by file: {str(e)}")
            return [{"error": str(e)}]
