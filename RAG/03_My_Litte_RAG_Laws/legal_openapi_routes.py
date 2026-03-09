from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from legal_retrieval_engine import LegalRetrievalEngine

# Create the API router
router = APIRouter()

# Define request/response models
class RetrieveParagraphRequest(BaseModel):
    law_name: str
    section_number: str


class ListLawsResponse(BaseModel):
    name: str
    abbreviation: str = ""
    section_count: int


class RetrieveParagraphResponse(BaseModel):
    law_name: str
    law_abbreviation: str = ""
    section_number: str
    content: str
    file_path: str


# Initialize the legal retrieval engine
legal_engine = LegalRetrievalEngine()


@router.post("/paragraph", response_model=RetrieveParagraphResponse,
             summary="Retrieve a specific legal paragraph",
             description="Retrieve the exact paragraph content for a given law and section number.")
async def api_retrieve_paragraph(request: RetrieveParagraphRequest) -> RetrieveParagraphResponse:
    """
    Retrieve a specific legal paragraph by law name and section number.

    Args:
        law_name: Name of the law (e.g., "Gesetz gegen Wettbewerbsbeschränkungen")
        section_number: Section number (e.g., "97" for §97)

    Returns:
        Dictionary with paragraph information or error if not found
    """
    try:
        result = legal_engine.retrieve_paragraph(
            law_name=request.law_name,
            section_number=request.section_number
        )

        if result is None or 'error' in result:
            error_msg = result.get('error', 'Paragraph not found') if result else 'Paragraph not found'
            raise HTTPException(status_code=404, detail=error_msg)

        return RetrieveParagraphResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/laws", response_model=List[ListLawsResponse],
            summary="List all available laws",
            description="Get information about all laws available in the system.")
async def api_list_laws() -> List[ListLawsResponse]:
    """
    List all available laws in the collection.

    Returns:
        List of law information including name and number of sections
    """
    try:
        results = legal_engine.list_laws()

        # Handle error case
        if results and 'error' in results[0]:
            raise HTTPException(status_code=500, detail=results[0]['error'])

        return [ListLawsResponse(**result) for result in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paragraph/{law_name}/{section_number}", response_model=RetrieveParagraphResponse,
            summary="Retrieve a specific legal paragraph by path parameters",
            description="Retrieve the exact paragraph content for a given law and section number using path parameters.")
async def api_retrieve_paragraph_path(law_name: str, section_number: str) -> RetrieveParagraphResponse:
    """
    Retrieve a specific legal paragraph by law name and section number using path parameters.

    Args:
        law_name: Name of the law (spaces should be replaced with underscores)
        section_number: Section number (e.g., "97" for §97)

    Returns:
        Dictionary with paragraph information or error if not found
    """
    try:
        # Replace underscores in law_name with spaces for proper lookup
        law_name_formatted = law_name.replace('_', ' ')

        result = legal_engine.retrieve_paragraph(
            law_name=law_name_formatted,
            section_number=section_number
        )

        if result is None or 'error' in result:
            error_msg = result.get('error', 'Paragraph not found') if result else 'Paragraph not found'
            raise HTTPException(status_code=404, detail=error_msg)

        return RetrieveParagraphResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


