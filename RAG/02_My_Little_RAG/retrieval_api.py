from mcp.server.fastmcp import FastMCP
from retrieval_engine import RetrievalEngine

# Initialize the retrieval engine
retrieval_engine = RetrievalEngine()

# Initialize the MCP server
mcp = FastMCP("Qdrant Retrieval Service")

@mcp.tool()
def search(
    query: str,
    top_k: int = 10,
    collection_name: str = None
) -> list[dict]:
    """
    General hybrid search across all documents in the Qdrant collection.

    Args:
        query: The search query text
        top_k: Number of top results to return (default 10)
        collection_name: Name of the collection to search in (defaults to main collection)

    Returns:
        List of search results with rank, score, payload, and content
    """
    return retrieval_engine.search(query, top_k, collection_name)

@mcp.tool()
def search_by_file(
    query: str,
    file_name: str,
    top_k: int = 10,
    collection_name: str = None
) -> list[dict]:
    """
    Hybrid search within a specific file in the Qdrant collection.

    Args:
        query: The search query text
        file_name: The specific file to search within
        top_k: Number of top results to return (default 10)
        collection_name: Name of the collection to search in (defaults to main collection)

    Returns:
        List of search results with rank, score, payload, and content from the specified file
    """
    return retrieval_engine.search_by_file(query, file_name, top_k, collection_name)

@mcp.tool()
def list_collections() -> list[dict]:
    """
    List all available Qdrant collections.

    Returns:
        List of collection information including name and point count
    """
    return retrieval_engine.list_collections()

@mcp.tool()
def text_search(
    query: str,
    top_k: int = 10,
    collection_name: str = None
) -> list[dict]:
    """
    Text-only search across all documents in knowledge base
    Performs a keyword-based search using sparse vectors across all documents in a specified Qdrant collection to find the most relevant content.

    Args:
        query: The search query text
        top_k: Number of top results to return (default 10)
        collection_name: Name of the collection to search in (defaults to main collection)

    Returns:
        List of search results with rank, score, content, file name and collection name
    """
    return retrieval_engine.text_search(query, top_k, collection_name)

@mcp.tool()
def text_search_by_file(
    query: str,
    file_name: str,
    top_k: int = 10,
    collection_name: str = None
) -> list[dict]:
    """
    Text-only search within a specific document
    Performs a keyword-based search using sparse vectors within a specific file to find relevant content from that document only.

    Args:
        query: The search query text
        file_name: The specific file to search within
        top_k: Number of top results to return (default 10)
        collection_name: Name of the collection to search in (defaults to main collection)

    Returns:
        List of search results with rank, score, content, file name and collection name from the specified file
    """
    return retrieval_engine.text_search_by_file(query, file_name, top_k, collection_name)

# Create the FastAPI app for the streamable HTTP transport
app = mcp.streamable_http_app()

# Run the server when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "retrieval_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
