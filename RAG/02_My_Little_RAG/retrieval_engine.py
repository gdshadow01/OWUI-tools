import os
import hashlib
import math
from typing import Optional, List, Dict, Any
from collections import Counter
import re

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    TextIndexParams,
    Filter,
    FieldCondition,
    MatchText,
    MatchValue,
    FilterSelector,
    SparseVectorParams,
    SparseVector,
    QueryRequest,
    Fusion,
    RecommendInput
)
from qdrant_client.models import Query
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

# Configuration
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
#OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://llamacpp.legally-berlin.de/v1")
OPENAI_BASE_URL = os.getenv("OPENAI_RETRIVAL_URL")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

# Sparse vector configuration
SPARSE_INDEX_SPACE = int(os.getenv("SPARSE_INDEX_SPACE", 100000))  # Same as previous hash space

def stable_term_index(term: str, mod=SPARSE_INDEX_SPACE) -> int:
    """Stable, cross-process deterministic hash -> index."""
    # md5 returns hex; convert to int
    h = hashlib.md5(term.encode('utf-8')).hexdigest()
    return int(h, 16) % mod

def generate_sparse_vector(text):
    """Generate a sparse vector representation of text with improved term weighting"""
    if not text or not text.strip():
        return SparseVector(indices=[], values=[])

    # Enhanced tokenization to handle German characters and special symbols including legal notation like §98
    # This pattern captures:
    # 1. § followed by optional whitespace and numbers (e.g., "§ 98", "§98") as single tokens
    # 2. Words that may contain special symbols
    # 3. Numbers
    # 4. Other word-like tokens
    tokens = re.findall(r'(?:§\s*\d+|[a-zA-ZäöüÄÖÜß§]+\d*|\d+[a-zA-ZäöüÄÖÜß§]*|[a-zA-ZäöüÄÖÜß§]+)', text.lower())

    # Keep only terms with length >= 2 (no stopwords filtering to match ingestion service)
    filtered_tokens = [token for token in tokens if len(token) >= 2]

    if not filtered_tokens:
        return SparseVector(indices=[], values=[])

    # Count term frequencies
    tf = Counter(filtered_tokens)

    # Calculate TF-IDF like weighting to give more importance to rare terms
    # For now, using a simple approach where more frequent terms in the document get
    # a boost, but we also want to avoid overwhelming very common terms
    total_terms = len(filtered_tokens)
    index_values = {}

    # Create indices based on stable hash of terms (to maintain consistency)
    # Handle potential hash collisions by summing frequencies
    for term, freq in tf.items():
        # Create a stable hash-based index for the term
        term_hash = stable_term_index(term)  # Limit index space

        # Apply a log-based weight to reduce the impact of very frequent terms
        # This is a simplified TF-IDF approach
        log_weight = 1 + math.log(freq)  # Adding 1 to avoid log(0)

        # If index already exists, sum the weights (handle hash collisions)
        if term_hash in index_values:
            index_values[term_hash] += log_weight
        else:
            index_values[term_hash] = log_weight

    # Convert to lists and sort by indices for Qdrant sparse vector format
    if index_values:
        # Sort by indices to maintain consistent order
        sorted_items = sorted(index_values.items())
        indices, values = zip(*sorted_items)
        indices = list(indices)
        values = list(values)

        # Apply min-max normalization to ensure consistent value range
        min_val, max_val = min(values), max(values)
        if max_val > min_val:
            # Normalize to [0, 1] range
            values = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            # If all values are the same, set them to 1.0
            values = [1.0 for _ in values]
    else:
        indices = []
        values = []

    return SparseVector(indices=indices, values=values)

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
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.embedding_func = get_embedding_function()

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
        collection_name: str = None
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
            # Use the specified collection or default to the main collection
            collection_to_search = collection_name if collection_name else QDRANT_COLLECTION

            # Get embedding function and create dense vector
            dense_vector = self.embedding_func.embed_query(query)

            # Create sparse vector from the query
            sparse_vector = generate_sparse_vector(query)

            # Attempt Qdrant's native hybrid search using the query endpoint with fusion
            try:
                # Use Qdrant's native hybrid search capability with Prefetch
                from qdrant_client import models

                search_result = self.client.query_points(
                    collection_name=collection_to_search,
                    prefetch=[
                        models.Prefetch(
                            query=dense_vector,
                            using="dense",
                            limit=top_k * 2,
                        ),
                        models.Prefetch(
                            query=sparse_vector,  # Pass SparseVector object directly
                            using="sparse",
                            limit=top_k * 3,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                    with_payload=True
                ).points  # Access the points from the response

            except Exception as e:
                print(f"Native hybrid search failed: {e}")
                # Fall back to simple dense vector search only
                search_result = self.client.query_points(
                    collection_name=collection_to_search,
                    query=dense_vector,
                    query_filter=None,
                    limit=top_k,
                    with_payload=True
                ).points  # Access the points from the response

            # Format results - return only essential information
            results = [
                {
                    "rank": idx + 1,
                    "score": point.score,
                    "content": point.payload.get("text", "") if point.payload else "",  # Use "text" from payload (now contains full paragraph)
                    "file_name": point.payload.get("source", "") if point.payload else "",  # Include file name
                    "collection_name": collection_to_search  # Include collection name in the result
                }
                for idx, point in enumerate(search_result)
            ]

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
        collection_name: str = None
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
            # Use the specified collection or default to the main collection
            collection_to_search = collection_name if collection_name else QDRANT_COLLECTION

            # Get embedding function and create dense vector
            dense_vector = self.embedding_func.embed_query(query)

            # Create sparse vector from the query
            sparse_vector = generate_sparse_vector(query)

            # Create a filter to limit search to the specific file
            file_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",  # Using the correct field name from payload
                        match=MatchValue(value=file_name)
                    )
                ]
            )

            # Attempt Qdrant's native hybrid search using the query endpoint with fusion and filter
            try:
                # Use Qdrant's native hybrid search capability with file filter
                from qdrant_client import models

                search_result = self.client.query_points(
                    collection_name=collection_to_search,
                    prefetch=[
                        models.Prefetch(
                            query=dense_vector,
                            using="dense",
                            limit=top_k * 2,
                            filter=file_filter,
                        ),
                        models.Prefetch(
                            query=sparse_vector,  # Pass SparseVector object directly
                            using="sparse",
                            limit=top_k * 3,
                            filter=file_filter,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                    with_payload=True
                ).points  # Access the points from the response

            except Exception as e:
                print(f"Native hybrid search failed: {e}")
                # Fall back to simple dense vector search with file filter only
                search_result = self.client.query_points(
                    collection_name=collection_to_search,
                    query=dense_vector,
                    limit=top_k,
                    with_payload=True
                ).points  # Access the points from the response

            # Format results - return only essential information
            results = [
                {
                    "rank": idx + 1,
                    "score": point.score,
                    "content": point.payload.get("text", "") if point.payload else "",  # Use "text" from payload (now contains full paragraph)
                    "file_name": point.payload.get("source", "") if point.payload else "",  # Include file name
                    "collection_name": collection_to_search  # Include collection name in the result
                }
                for idx, point in enumerate(search_result)
            ]

            return results

        except Exception as e:
            print(f"Error during search by file: {str(e)}")
            return [{"error": str(e)}]

    def text_search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        collection_name: str = None
    ) -> List[dict]:
        """
        Text-only search using sparse vectors across all documents in the Qdrant collection.
        This performs keyword-based search using sparse vector representation of the query.

        Args:
            query: The search query text
            top_k: Number of top results to return (default 10)
            min_score: Minimum similarity score for results (default 0.0)
            collection_name: Name of the collection to search in (defaults to main collection)

        Returns:
            List of search results with id, score, payload, and content
        """
        try:
            # Use the specified collection or default to the main collection
            collection_to_search = collection_name if collection_name else QDRANT_COLLECTION

            # Create sparse vector from the query
            sparse_vector = generate_sparse_vector(query)

            # Perform search using only the sparse vector
            # For sparse vector search, we need to pass the sparse vector in the correct format
            # The sparse vector should be passed with the proper structure
            search_result = self.client.query_points(
                collection_name=collection_to_search,
                query=sparse_vector,
                using="sparse",  # Specify that we want to use the sparse vector configuration
                limit=top_k,
                with_payload=True
            ).points  # Access the points from the response

            # Format results - return only essential information
            results = [
                {
                    "rank": idx + 1,
                    "score": point.score,
                    "content": point.payload.get("text", "") if point.payload else "",  # Use "text" from payload (now contains full paragraph)
                    "file_name": point.payload.get("source", "") if point.payload else "",  # Include file name
                    "collection_name": collection_to_search  # Include collection name in the result
                }
                for idx, point in enumerate(search_result)
            ]

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
        collection_name: str = None
    ) -> List[dict]:
        """
        Text-only search using sparse vectors within a specific file in the Qdrant collection.
        This performs keyword-based search using sparse vector representation of the query.

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
            # Use the specified collection or default to the main collection
            collection_to_search = collection_name if collection_name else QDRANT_COLLECTION

            # Create sparse vector from the query
            sparse_vector = generate_sparse_vector(query)

            # Create a filter to limit search to the specific file
            file_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",  # Using the correct field name from payload
                        match=MatchValue(value=file_name)
                    )
                ]
            )

            # Perform search using only the sparse vector with file filter
            search_result = self.client.query_points(
                collection_name=collection_to_search,
                prefetch=[
                    models.Prefetch(
                        query=sparse_vector,
                        using="sparse",
                        limit=top_k,
                        filter=file_filter,
                    ),
                ],
                limit=top_k,
                with_payload=True
            ).points  # Access the points from the response

            # Format results - return only essential information
            results = [
                {
                    "rank": idx + 1,
                    "score": point.score,
                    "content": point.payload.get("text", "") if point.payload else "",  # Use "text" from payload (now contains full paragraph)
                    "file_name": point.payload.get("source", "") if point.payload else "",  # Include file name
                    "collection_name": collection_to_search  # Include collection name in the result
                }
                for idx, point in enumerate(search_result)
            ]

            return results

        except Exception as e:
            print(f"Error during text search by file: {str(e)}")
            return [{"error": str(e)}]
