import os
from fastapi import FastAPI
from openapi_routes import router as openapi_router

# Get the transport type from environment variable
TRANSPORT_TYPE = os.getenv("TRANSPORT_TYPE", "openapi").lower()  # Default to openapi for legal api

# Import required modules
from legal_retrieval_engine import LegalRetrievalEngine

# Conditionally create and run the appropriate server based on TRANSPORT_TYPE
if TRANSPORT_TYPE == "mcp":
    # Only MCP mode: Run MCP server directly
    from mcp.server.fastmcp import FastMCP

    # Initialize the legal retrieval engine
    legal_engine = LegalRetrievalEngine()
    print(f"Legal Paragraph Retrieval Service initialized with:")

    # Initialize the MCP server
    mcp = FastMCP("Legal Paragraph Retrieval Service")

    @mcp.tool()
    def retrieve_paragraph(
        law_name: str,
        section_number: str
    ) -> dict:
        """
        Retrieve a specific legal paragraph by law name and section number
        Retrieves the exact paragraph content for a given law and section number.

        Args:
            law_name: Name of the law (e.g., "Gesetz gegen Wettbewerbsbeschränkungen")
            section_number: Section number (e.g., "97" for §97)

        Returns:
            Dictionary with paragraph information or None if not found
        """
        return legal_engine.retrieve_paragraph(law_name, section_number)

    @mcp.tool()
    def list_laws() -> list[dict]:
        """
        List all available laws in the collection
        Returns information about all the laws available in the system.

        Returns:
            List of law information including name and number of sections
        """
        return legal_engine.list_laws()

    # Run the MCP server directly
    app = mcp.streamable_http_app()
elif TRANSPORT_TYPE == "openapi":
    # Only OpenAPI mode: Run FastAPI server with OpenAPI endpoints
    app = FastAPI(
        title="Legal Paragraph Retrieval Service",
        description="An API service to retrieve specific legal paragraphs (§) from German laws",
        version="1.0.0"
    )

    # Initialize the legal retrieval engine
    legal_engine = LegalRetrievalEngine()
    print(f"Legal Paragraph Retrieval Service initialized")

    # Include OpenAPI routes
    app.include_router(
        openapi_router,
        prefix="/api",
        tags=["legal_paragraphs"]
    )

else:
    # Default to OpenAPI mode when both is specified
    app = FastAPI(
        title="Legal Paragraph Retrieval Service",
        description="An API service to retrieve specific legal paragraphs (§) from German laws",
        version="1.0.0"
    )

    # Initialize the legal retrieval engine
    legal_engine = LegalRetrievalEngine()
    print(f"Legal Paragraph Retrieval Service initialized")

    # Include OpenAPI routes
    app.include_router(
        openapi_router,
        prefix="/api",
        tags=["legal_paragraphs"]
    )

    @app.get("/")
    async def root():
        return {
            "message": "Legal Paragraph Retrieval Service - OpenAPI Mode",
            "available_services": ["OpenAPI"],
            "docs": "/docs",
            "api_endpoint": "/api"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )