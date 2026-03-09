from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from retrieval_engine import RetrievalEngine

# Create the API router
router = APIRouter()

# Define request/response models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    collection_name: str


class SearchByFileRequest(BaseModel):
    query: str
    file_name: str = Field(..., description="The exact full path of the specific file to search within (required).")
    top_k: int = 10
    collection_name: str


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    collection_name: str


class TextSearchByFileRequest(BaseModel):
    query: str
    file_name: str = Field(..., description="The exact full path of the specific file to search within (required).")
    top_k: int = 10
    collection_name: str


class CollectionInfo(BaseModel):
    name: str
    point_count: int
    vectors_count: int
    indexed_vectors_count: Optional[int] = 0
    has_dense_vectors: Optional[bool] = False
    has_sparse_vectors: Optional[bool] = False


class SearchResult(BaseModel):
    rank: int
    score: float
    content: str
    file_name: str
    collection_name: Optional[str] = None


# Initialize the retrieval engine
retrieval_engine = RetrievalEngine()


@router.post("/search", response_model=List[SearchResult], summary="Comprehensive hybrid search across all documents in knowledge base",
             description="Performs a semantic and keyword hybrid search across all documents in the specified Qdrant collection to find the most relevant content. This endpoint combines dense vector search (semantic meaning) with sparse vector search (keyword matching) for superior retrieval accuracy. Use this endpoint when you need to find general information across all documents in the knowledge base.")
async def api_search(request: SearchRequest) -> List[SearchResult]:
    try:
        results = retrieval_engine.search(
            query=request.query,
            top_k=request.top_k,
            collection_name=request.collection_name
        )

        # Handle error case
        if results and 'error' in results[0]:
            raise HTTPException(status_code=500, detail=results[0]['error'])

        return [SearchResult(**result) for result in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search_by_file", response_model=List[SearchResult], summary="Targeted hybrid search within a specific document",
             description="Performs a semantic and keyword hybrid search within a specific file to find relevant content from that document only. This endpoint combines dense vector search (semantic meaning) with sparse vector search (keyword matching) for superior retrieval accuracy. Use this endpoint when you need to find information within a particular document and know the exact filename.")
async def api_search_by_file(request: SearchByFileRequest) -> List[SearchResult]:
    try:
        results = retrieval_engine.search_by_file(
            query=request.query,
            file_name=request.file_name,
            top_k=request.top_k,
            collection_name=request.collection_name
        )

        # Handle error case
        if results and 'error' in results[0]:
            raise HTTPException(status_code=500, detail=results[0]['error'])

        return [SearchResult(**result) for result in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections", response_model=List[CollectionInfo], summary="List all available knowledge collections and their metadata",
             description="Returns comprehensive information about all available knowledge collections in Qdrant, including document counts, vector configurations, and collection statistics. This endpoint is essential for understanding what data is available for searching.")
async def list_collections_api() -> List[CollectionInfo]:
    try:
        results = retrieval_engine.list_collections()

        # Handle error case
        if results and 'error' in results[0]:
            raise HTTPException(status_code=500, detail=results[0]['error'])

        return [CollectionInfo(**result) for result in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text_search", response_model=List[SearchResult], summary="Keyword-focused text search across all documents in knowledge base",
             description="Performs a keyword-based search using sparse vectors across all documents in the specified Qdrant collection to find the most relevant content. This endpoint focuses on exact keyword matching rather than semantic meaning, making it ideal for finding specific terms, phrases, or literal matches.")
async def api_text_search(request: TextSearchRequest) -> List[SearchResult]:
    try:
        results = retrieval_engine.text_search(
            query=request.query,
            top_k=request.top_k,
            collection_name=request.collection_name
        )

        # Handle error case
        if results and 'error' in results[0]:
            raise HTTPException(status_code=500, detail=results[0]['error'])

        return [SearchResult(**result) for result in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text_search_by_file", response_model=List[SearchResult], summary="Keyword-focused text search within a specific document",
             description="Performs a keyword-based search using sparse vectors within a specific file to find relevant content from that document only. This endpoint focuses on exact keyword matching rather than semantic meaning, making it ideal for finding specific terms, phrases, or literal matches within a particular document.")
async def api_text_search_by_file(request: TextSearchByFileRequest) -> List[SearchResult]:
    try:
        results = retrieval_engine.text_search_by_file(
            query=request.query,
            file_name=request.file_name,
            top_k=request.top_k,
            collection_name=request.collection_name
        )

        # Handle error case
        if results and 'error' in results[0]:
            raise HTTPException(status_code=500, detail=results[0]['error'])

        return [SearchResult(**result) for result in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


