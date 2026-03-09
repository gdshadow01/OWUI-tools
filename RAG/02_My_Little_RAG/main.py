import os
from fastapi import FastAPI
from openapi_routes import router as openapi_router

# Get the transport type from environment variable
TRANSPORT_TYPE = os.getenv("TRANSPORT_TYPE", "both").lower()

# Conditionally create and run the appropriate server based on TRANSPORT_TYPE
if TRANSPORT_TYPE == "mcp":
    # Only MCP mode: Run MCP server directly
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
        Comprehensive hybrid search across all documents in the knowledge base

        Performs a semantic and keyword hybrid search across all documents in the specified Qdrant collection to find the most relevant content.
        This tool combines dense vector search (semantic meaning) with sparse vector search (keyword matching) for superior retrieval accuracy.

        Use this tool when you need to find general information across all documents in the knowledge base.
        For searching within a specific document, use search_by_file instead.

        Args:
            query: The search query text containing the information you're looking for.
                  This can be a question, statement, or keywords related to what you need.
                  Example: "What are the legal requirements for data protection in Germany?"

            top_k: Number of top results to return (default 10, minimum 1, maximum 100).
                   Controls how many results are returned. Use lower values for focused results,
                   higher values when exploring broader topics.
                   Example: top_k=5 for top 5 most relevant results.

            collection_name: Name of the Qdrant collection to search in (optional).
                             If not specified, searches in the default collection.
                             Use list_collections tool to see available collections.
                             Example: collection_name="legal_documents" to search a specific collection.

        Returns:
            List of search results, each containing:
              - rank: Position in the ranked results (1 is most relevant)
              - score: Relevance score (for metadata purposes, not used as decision signal)
              - content: The actual text content of the matching passage/paragraph
              - file_name: Name of the source file where the content was found
              - collection_name: Name of the collection where the result was found

        Error Handling:
            - If collection doesn't exist: Returns error message suggesting to check collection names with list_collections
            - If query is empty: Returns error message indicating query parameter is required
            - If connection fails: Returns error message about connection issues
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
        Targeted hybrid search within a specific document in the knowledge base

        Performs a semantic and keyword hybrid search within a specific file to find relevant content from that document only.
        This tool combines dense vector search (semantic meaning) with sparse vector search (keyword matching) for superior retrieval accuracy.

        Use this tool when you need to find information within a particular document.
        You must know the exact filename (with extension) where you expect to find the information.
        Use list_collections tool to see available files in each collection.

        Args:
            query: The search query text containing the information you're looking for.
                  This can be a question, statement, or keywords related to what you need.
                  Example: "What are the penalties for GDPR violations?"

            file_name: The exact name of the specific file to search within (required).
                      Include the file extension. The file must exist in the specified collection.
                      Use list_collections tool to see available files.
                      Example: file_name="gdpr_policy.pdf" or file_name="contract_agreement.docx"

            top_k: Number of top results to return (default 10, minimum 1, maximum 100).
                   Controls how many results are returned from within the specified document.
                   Use lower values for focused results, higher values when exploring broader topics within the document.
                   Example: top_k=3 for top 3 most relevant results within the file.

            collection_name: Name of the Qdrant collection to search in (optional).
                             If not specified, searches in the default collection.
                             Use list_collections tool to see available collections.
                             Example: collection_name="legal_documents" to search in a specific collection.

        Returns:
            List of search results, each containing:
              - rank: Position in the ranked results (1 is most relevant)
              - score: Relevance score (for metadata purposes, not used as decision signal)
              - content: The actual text content of the matching passage/paragraph
              - file_name: Name of the source file where the content was found (will match input parameter)
              - collection_name: Name of the collection where the result was found

        Error Handling:
            - If file doesn't exist: Returns error message indicating the file wasn't found in the collection
            - If collection doesn't exist: Returns error message suggesting to check collection names with list_collections
            - If query is empty: Returns error message indicating query parameter is required
            - If connection fails: Returns error message about connection issues
        """
        return retrieval_engine.search_by_file(query, file_name, top_k, collection_name)

    @mcp.tool()
    def list_collections() -> list[dict]:
        """
        List all available knowledge collections and their metadata

        Returns comprehensive information about all available knowledge collections in Qdrant,
        including document counts, vector configurations, and collection statistics.
        This tool is essential for understanding what data is available for searching.

        Use this tool before performing searches to:
        1. Discover what collections are available
        2. Check how much data is in each collection (point_count)
        3. Understand the vector configuration of each collection
        4. Determine which collection to search in for specific queries

        Args:
            This tool takes no parameters.

        Returns:
            List of collection objects, each containing:
              - name: The name of the collection (use this in other search tools)
              - point_count: Number of documents/chunks stored in the collection
              - vectors_count: Total number of vectors stored
              - indexed_vectors_count: Number of vectors that have been indexed for faster search
              - has_dense_vectors: Boolean indicating if the collection supports dense vector search (semantic)
              - has_sparse_vectors: Boolean indicating if the collection supports sparse vector search (keyword)

            Example return:
            [
              {
                "name": "legal_documents",
                "point_count": 1542,
                "vectors_count": 3084,
                "indexed_vectors_count": 3084,
                "has_dense_vectors": True,
                "has_sparse_vectors": True
              },
              {
                "name": "tech_manuals",
                "point_count": 896,
                "vectors_count": 1792,
                "indexed_vectors_count": 1792,
                "has_dense_vectors": True,
                "has_sparse_vectors": True
              }
            ]

        Error Handling:
            - If Qdrant server is unavailable: Returns error message about connection issues
            - If no collections exist: Returns empty list
        """
        return retrieval_engine.list_collections()

    @mcp.tool()
    def text_search(
        query: str,
        top_k: int = 10,
        collection_name: str = None
    ) -> list[dict]:
        """
        Keyword-focused text search across all documents in the knowledge base

        Performs a keyword-based search using sparse vectors across all documents in the specified Qdrant collection to find the most relevant content.
        This tool focuses on exact keyword matching rather than semantic meaning, making it ideal for finding specific terms, phrases, or literal matches.

        Use this tool when you need to find exact keywords, technical terms, specific phrases, or when semantic similarity isn't as important.
        For more semantic understanding, use the general 'search' tool instead.

        Args:
            query: The search query text containing the keywords or phrases you're looking for.
                  This should contain the exact terms you want to match in the documents.
                  Example: "GDPR Article 5" or "Section 2.3" or "penalty amount"

            top_k: Number of top results to return (default 10, minimum 1, maximum 100).
                   Controls how many results are returned. Use lower values for focused results,
                   higher values when exploring broader topics.
                   Example: top_k=5 for top 5 most relevant results.

            collection_name: Name of the Qdrant collection to search in (optional).
                             If not specified, searches in the default collection.
                             Use list_collections tool to see available collections.
                             Example: collection_name="technical_docs" to search in a specific collection.

        Returns:
            List of search results, each containing:
              - rank: Position in the ranked results (1 is most relevant)
              - score: Relevance score (for metadata purposes, not used as decision signal)
              - content: The actual text content of the matching passage/paragraph
              - file_name: Name of the source file where the content was found
              - collection_name: Name of the collection where the result was found

        Error Handling:
            - If collection doesn't exist: Returns error message suggesting to check collection names with list_collections
            - If query is empty: Returns error message indicating query parameter is required
            - If connection fails: Returns error message about connection issues
            - If collection doesn't support sparse vectors: Returns error message explaining the limitation
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
        Keyword-focused text search within a specific document in the knowledge base

        Performs a keyword-based search using sparse vectors within a specific file to find relevant content from that document only.
        This tool focuses on exact keyword matching rather than semantic meaning, making it ideal for finding specific terms, phrases, or literal matches within a particular document.

        Use this tool when you need to find exact keywords, technical terms, specific phrases, or when semantic similarity isn't as important within a specific document.
        You must know the exact filename (with extension) where you expect to find the information.
        Use list_collections tool to see available files in each collection.

        Args:
            query: The search query text containing the keywords or phrases you're looking for.
                  This should contain the exact terms you want to match in the document.
                  Example: "GDPR Article 5" or "Section 2.3" or "penalty amount"

            file_name: The exact name of the specific file to search within (required).
                      Include the file extension. The file must exist in the specified collection.
                      Use list_collections tool to see available files.
                      Example: file_name="gdpr_policy.pdf" or file_name="contract_agreement.docx"

            top_k: Number of top results to return (default 10, minimum 1, maximum 100).
                   Controls how many results are returned from within the specified document.
                   Use lower values for focused results, higher values when exploring broader topics within the document.
                   Example: top_k=3 for top 3 most relevant results within the file.

            collection_name: Name of the Qdrant collection to search in (optional).
                             If not specified, searches in the default collection.
                             Use list_collections tool to see available collections.
                             Example: collection_name="technical_docs" to search in a specific collection.

        Returns:
            List of search results, each containing:
              - rank: Position in the ranked results (1 is most relevant)
              - score: Relevance score (for metadata purposes, not used as decision signal)
              - content: The actual text content of the matching passage/paragraph
              - file_name: Name of the source file where the content was found (will match input parameter)
              - collection_name: Name of the collection where the result was found

        Error Handling:
            - If file doesn't exist: Returns error message indicating the file wasn't found in the collection
            - If collection doesn't exist: Returns error message suggesting to check collection names with list_collections
            - If query is empty: Returns error message indicating query parameter is required
            - If connection fails: Returns error message about connection issues
            - If collection doesn't support sparse vectors: Returns error message explaining the limitation
        """
        return retrieval_engine.text_search_by_file(query, file_name, top_k, collection_name)

    # Run the MCP server directly
    app = mcp.streamable_http_app()
elif TRANSPORT_TYPE == "openapi":
    # Only OpenAPI mode: Run FastAPI server with OpenAPI endpoints
    app = FastAPI(
        title="My Little RAG Retrieval Service",
        description="A retrieval service with hybrid search capabilities using dense and sparse vectors",
        version="1.0.0"
    )

    # Include OpenAPI routes
    app.include_router(
        openapi_router,
        prefix="/api",
        tags=["retrieval"]
    )

else:
    # Default to OpenAPI mode when both is specified
    app = FastAPI(
        title="My Little RAG Retrieval Service",
        description="A retrieval service with hybrid search capabilities using dense and sparse vectors",
        version="1.0.0"
    )

    # Include OpenAPI routes
    app.include_router(
        openapi_router,
        prefix="/api",
        tags=["retrieval"]
    )

    @app.get("/")
    async def root():
        return {
            "message": "My Little RAG Retrieval Service - OpenAPI Mode",
            "available_services": ["OpenAPI"],
            "docs": "/docs"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )