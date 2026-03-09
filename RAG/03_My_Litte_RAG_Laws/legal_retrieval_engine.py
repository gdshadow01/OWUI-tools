import os
from typing import Optional, List, Dict, Any

from legal_parser import LegalParagraph, load_legal_paragraphs

# Configuration for legal API
# Document directory
LEGAL_DOCUMENTS_DIR = os.getenv("LEGAL_DOCUMENTS_DIR", "input")


class LegalRetrievalEngine:
    """Class that encapsulates legal paragraph retrieval functionality"""

    def __init__(self):
        self.legal_parser = None
        self._load_legal_paragraphs()

    def _load_legal_paragraphs(self):
        """Load legal paragraphs from documents directory"""
        try:
            self.legal_paragraphs = load_legal_paragraphs(LEGAL_DOCUMENTS_DIR)
            print(f"Loaded {len(self.legal_paragraphs)} legal paragraphs")
        except Exception as e:
            print(f"Error loading legal paragraphs: {e}")
            self.legal_paragraphs = []

    def retrieve_paragraph(self, law_name: str, section_number: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific legal paragraph by law name and section number.

        Args:
            law_name: Name of the law (e.g., "Gesetz gegen Wettbewerbsbeschränkungen")
            section_number: Section number (e.g., "97" for §97)

        Returns:
            Dictionary with paragraph information or None if not found
        """
        try:
            # First, try exact matching using our parser
            for paragraph in self.legal_paragraphs:
                # Compare law names and abbreviations (case-insensitive, normalize spaces)
                paragraph_law = paragraph.law_name.lower().replace('  ', ' ').strip()
                paragraph_abbreviation = paragraph.law_abbreviation.lower().replace('  ', ' ').strip()
                search_law = law_name.lower().replace('  ', ' ').strip()

                # Compare section numbers (normalize)
                paragraph_section = paragraph.section_number.replace('§', '').strip()
                search_section = section_number.replace('§', '').strip()

                # Check if either the full law name or the abbreviation matches
                law_match = (paragraph_law == search_law or paragraph_abbreviation == search_law)

                if law_match and paragraph_section == search_section:
                    return {
                        "law_name": paragraph.law_name,
                        "law_abbreviation": paragraph.law_abbreviation,  # Include the abbreviation in the result
                        "section_number": paragraph.section_number,
                        "content": paragraph.content,
                        "file_path": paragraph.file_path
                    }

            # If exact match not found, try fuzzy matching
            for paragraph in self.legal_paragraphs:
                # Partial matching for law name and abbreviation
                paragraph_law = paragraph.law_name.lower()
                paragraph_abbreviation = paragraph.law_abbreviation.lower()
                search_law = law_name.lower()

                paragraph_section = paragraph.section_number.replace('§', '').strip()
                search_section = section_number.replace('§', '').strip()

                # Check if search law name is contained in paragraph law name or abbreviation, or vice versa
                law_match = (search_law in paragraph_law) or (paragraph_law in search_law) or \
                           (search_law in paragraph_abbreviation) or (paragraph_abbreviation in search_law)
                section_match = paragraph_section == search_section

                if law_match and section_match:
                    return {
                        "law_name": paragraph.law_name,
                        "law_abbreviation": paragraph.law_abbreviation,  # Include the abbreviation in the result
                        "section_number": paragraph.section_number,
                        "content": paragraph.content,
                        "file_path": paragraph.file_path
                    }

            return None

        except Exception as e:
            print(f"Error retrieving paragraph: {str(e)}")
            return {"error": str(e)}

    def list_laws(self) -> List[Dict[str, Any]]:
        """
        List all available laws in the collection.

        Returns:
            List of law information including name and number of sections
        """
        try:
            # Get unique law names from loaded paragraphs
            law_stats = {}
            for paragraph in self.legal_paragraphs:
                law_name = paragraph.law_name
                law_abbreviation = paragraph.law_abbreviation

                # Use the law name as the key, but include abbreviation as well
                key = law_name
                if key not in law_stats:
                    law_stats[key] = {"name": law_name, "abbreviation": law_abbreviation, "section_count": 0}
                law_stats[key]["section_count"] += 1

            return list(law_stats.values())
        except Exception as e:
            print(f"Error listing laws: {str(e)}")
            return [{"error": str(e)}]