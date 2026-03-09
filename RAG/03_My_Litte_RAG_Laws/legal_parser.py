import re
import os
from typing import List, Dict, Optional
from pathlib import Path
import markdown
from dataclasses import dataclass

@dataclass
class LegalParagraph:
    law_name: str
    law_abbreviation: str
    section_number: str
    content: str
    section_title: str = ""
    file_path: str = ""


class LegalDocumentParser:
    """Parses legal documents to extract paragraphs (§) with their associated law names."""
    
    def __init__(self, documents_dir: str = "input"):
        self.documents_dir = documents_dir
        self.parsed_paragraphs: List[LegalParagraph] = []
        
    def parse_all_documents(self) -> List[LegalParagraph]:
        """Parse all markdown documents in the documents directory."""
        self.parsed_paragraphs = []
        
        # Find all markdown files in the documents directory
        for file_path in Path(self.documents_dir).glob("*.md"):
            self._parse_document(file_path)
            
        return self.parsed_paragraphs
    
    def _parse_document(self, file_path: Path):
        """Parse a single legal document file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract the law name and abbreviation from the standardized header
            law_name, law_abbreviation = self._extract_law_info(content)

            # Extract paragraphs using regex
            sections = self._extract_sections(content)

            for section_number, section_title, section_content in sections:
                paragraph = LegalParagraph(
                    law_name=law_name,
                    law_abbreviation=law_abbreviation,
                    section_number=section_number,
                    content=section_content,
                    section_title=section_title,
                    file_path=str(file_path)
                )
                self.parsed_paragraphs.append(paragraph)

        except Exception as e:
            print(f"Error parsing document {file_path}: {e}")
    
    def _extract_law_info(self, content: str) -> tuple:
        """Extract law name and abbreviation from the standardized header format."""
        lines = content.split('\n')

        law_name = "Unknown Law"
        law_abbreviation = ""

        # Look for the standardized header format in the first 10 lines
        for line in lines[:10]:
            line = line.strip()

            # Extract law name from line like "# Gesetz: Verordnung über die Vergabe öffentlicher Aufträge"
            if line.startswith('# Gesetz:'):
                law_name = line.replace('# Gesetz:', '', 1).strip()

            # Extract abbreviation from line like "# Abkürzung: VgV"
            elif line.startswith('# Abkürzung:'):
                law_abbreviation = line.replace('# Abkürzung:', '', 1).strip()

        # If we couldn't find the standardized format, fallback to the previous method
        if law_name == "Unknown Law":
            for line in lines[:10]:
                line = line.strip()
                # Match markdown headings like "# Gesetz..." or "# Verordnung..."
                match = re.match(r'^#\s+(.+?)(?:\s+#|$)', line)
                if match:
                    law_name = match.group(1).strip()
                    break

                # Match lines that start with law abbreviations
                match = re.match(r'^([A-Z]{2,4})\s+(.+?)(?:\s+Ausfertigungsdatum:|Vollzitat:|$)', line)
                if match:
                    law_name = f"{match.group(1)} - {match.group(2).strip()}"
                    break

        return law_name, law_abbreviation
    
    def _extract_sections(self, content: str) -> List[tuple]:
        """Extract all sections (paragraphs) from the document."""
        sections = []

        # Split content into lines and identify sections that start with § or Artikel at beginning of line
        # Then reassemble the content for each section, ensuring we only start sections
        # when § or Artikel appears at the beginning of a line (optionally with leading whitespace)
        lines = content.split('\n')

        current_section_lines = []
        sections_data = []

        for line in lines:
            # Check if this line starts a new section (has § followed by number OR Artikel followed by number at the beginning)
            section_start_match = re.match(r'^(?:\s*§\s*\d+[a-zA-Z]?\b|\s*Artikel\s+\d+[a-zA-Z]?\b)', line, re.IGNORECASE)
            if section_start_match:
                # If we were collecting a previous section, save it
                if current_section_lines:
                    sections_data.append('\n'.join(current_section_lines))
                # Start collecting the new section
                current_section_lines = [line]
            else:
                # If this isn't a section start, add to current section if we're in one
                if current_section_lines:
                    current_section_lines.append(line)

        # Don't forget the last section
        if current_section_lines:
            sections_data.append('\n'.join(current_section_lines))

        # Process the collected sections
        for section_text in sections_data:
            section_text = section_text.strip()
            if not section_text:
                continue

            # Extract section number from the beginning of the section text
            # Check for both § and Artikel patterns
            section_num_match = re.match(r'^(?:\s*§\s*\d+[a-zA-Z]?\b|\s*Artikel\s+\d+[a-zA-Z]?\b)', section_text, re.IGNORECASE)
            if section_num_match:
                section_full_match = section_num_match.group(0).strip()
                # Extract just the number part after § or Artikel
                if section_full_match.lower().startswith('artikel'):
                    # Extract number after "Artikel"
                    number_match = re.search(r'\d+[a-zA-Z]?', section_full_match, re.IGNORECASE)
                    if number_match:
                        section_number = number_match.group(0)
                    else:
                        continue  # Skip if we can't extract number
                else:
                    # Extract number after §
                    section_number = section_full_match.replace('§', '').replace(' ', '').strip()
            else:
                continue  # Skip if we can't extract section number

            # The content is the full section text
            section_content = section_text

            # Set section_title to empty
            section_title = ""

            sections.append((section_number, section_title, section_content))

        return sections
    
    def find_paragraph(self, law_name: str, section_number: str) -> Optional[LegalParagraph]:
        """Find a specific paragraph by law name and section number."""
        for paragraph in self.parsed_paragraphs:
            # Compare law names (case-insensitive, normalize spaces)
            paragraph_law = paragraph.law_name.lower().replace('  ', ' ').strip()
            paragraph_abbreviation = paragraph.law_abbreviation.lower().replace('  ', ' ').strip()
            search_law = law_name.lower().replace('  ', ' ').strip()

            # Compare section numbers (normalize)
            paragraph_section = paragraph.section_number.replace('§', '').strip()
            search_section = section_number.replace('§', '').strip()

            # Check if either the full law name or the abbreviation matches
            law_match = (paragraph_law == search_law or paragraph_abbreviation.lower() == search_law)

            if law_match and paragraph_section == search_section:
                return paragraph

        return None
    
    def search_paragraphs_by_content(self, query: str) -> List[LegalParagraph]:
        """Search for paragraphs by content."""
        results = []
        query_lower = query.lower()

        for paragraph in self.parsed_paragraphs:
            if (query_lower in paragraph.content.lower() or
                query_lower in paragraph.section_title.lower() or
                query_lower in paragraph.law_name.lower() or
                query_lower in paragraph.law_abbreviation.lower()):
                results.append(paragraph)

        return results


def load_legal_paragraphs(documents_dir: str = "input") -> List[LegalParagraph]:
    """Load and parse all legal documents."""
    parser = LegalDocumentParser(documents_dir)
    return parser.parse_all_documents()


if __name__ == "__main__":
    # Example usage
    paragraphs = load_legal_paragraphs()
    print(f"Loaded {len(paragraphs)} legal paragraphs")

    # Example search
    for p in paragraphs[:5]:  # Show first 5 paragraphs
        print(f"Law: {p.law_name}, Abbreviation: {p.law_abbreviation}, Section: {p.section_number}, Title: {p.section_title}")