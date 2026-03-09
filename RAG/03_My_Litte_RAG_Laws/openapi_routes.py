from fastapi import APIRouter
from legal_openapi_routes import router as legal_router

# Create the API router and include legal-specific routes
router = APIRouter()

# Include legal API routes without prefix to provide clean endpoints for MCP/LLM
# This prevents duplicate tools being generated for the LLM
router.include_router(legal_router)


