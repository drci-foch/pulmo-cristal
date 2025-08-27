"""
PDF Extraction Module for pulmo-cristal package.

This module provides functionality to extract content from PDF files,
focusing on extracting text content, finding sections, and
processing donor documents.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import re
from .gds import ImprovedGazDuSangExtractor
from .hemodynamic import ImprovedHemodynamicExtractor

# Third-party imports
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Local imports
from .base import BaseExtractor
from .patterns import get_pattern_group


class PDFExtractor(BaseExtractor):
    """
    Extracts text content and sections from PDF documents.

    This class handles the basic PDF reading operations and text extraction,
    providing methods to access both the full text and specific page content.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, debug: bool = False):
        """
        Initialize the PDF extractor.

        Args:
            logger: Optional logger instance
            debug: Enable debug mode for verbose logging
        """
        super().__init__(logger=logger)
        self.debug = debug
        self._extracted_text = None
        self._page_texts = []

        # Check if PyPDF2 is available
        if PyPDF2 is None:
            self.log(
                "PyPDF2 is not installed. PDF extraction will not be available.",
                level=logging.ERROR,
            )

    def extract_from_pdf(self, pdf_path: Union[str, Path]) -> Tuple[str, List[str]]:
        """
        Extract text content from a PDF file.

        Args:
            pdf_path: Path to the PDF file (str or Path object)

        Returns:
            Tuple containing:
                - Full text content as a single string
                - List of text content per page

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            PermissionError: If the PDF file can't be accessed
            ValueError: If PyPDF2 is not installed
        """
        if PyPDF2 is None:
            raise ValueError("PyPDF2 is not installed. Cannot extract PDF content.")

        # Reset stored text
        self._extracted_text = ""
        self._page_texts = []

        # Convert to Path object for better path handling
        path_obj = Path(pdf_path) if isinstance(pdf_path, str) else pdf_path

        try:
            self.log(f"Extracting text from PDF: {path_obj}")

            with open(path_obj, "rb") as file:
                try:
                    reader = PyPDF2.PdfReader(file)
                    num_pages = len(reader.pages)
                    self.log(f"PDF contains {num_pages} pages")

                    # Extract text from each page
                    for i, page in enumerate(reader.pages):
                        try:
                            page_text = page.extract_text()
                            self._page_texts.append(page_text)
                            self.log(
                                f"Page {i + 1}/{num_pages}: Extracted {len(page_text)} characters"
                            )
                        except Exception as e:
                            self.log(
                                f"Error extracting text from page {i + 1}: {str(e)}",
                                level=logging.WARNING,
                            )
                            self._page_texts.append("")

                    # Combine all pages with page separators
                    self._extracted_text = "\n\n".join(self._page_texts)
                    self.log(
                        f"Total extracted text: {len(self._extracted_text)} characters"
                    )

                except PyPDF2.errors.PdfReadError as e:
                    self.log(
                        f"Error reading PDF structure: {str(e)}", level=logging.ERROR
                    )
                    raise
                except Exception as e:
                    self.log(
                        f"Unexpected error in PDF processing: {str(e)}",
                        level=logging.ERROR,
                    )
                    raise

        except FileNotFoundError:
            self.log(f"PDF file not found: {path_obj}", level=logging.ERROR)
            raise
        except PermissionError:
            self.log(
                f"Permission denied when accessing: {path_obj}", level=logging.ERROR
            )
            raise
        except Exception as e:
            self.log(f"Error opening PDF file: {str(e)}", level=logging.ERROR)
            raise

        return self._extracted_text, self._page_texts

    def get_text(self) -> str:
        """
        Get the extracted text content.

        Returns:
            Full text content as a string
        """
        if self._extracted_text is None:
            self.log(
                "No text has been extracted yet. Call extract_from_pdf first.",
                level=logging.WARNING,
            )
            return ""

        return self._extracted_text

    def get_page_text(self, page_index: int) -> str:
        """
        Get the text content of a specific page.

        Args:
            page_index: 0-based index of the page

        Returns:
            Text content of the specified page

        Raises:
            IndexError: If the page index is out of range
        """
        if not self._page_texts:
            self.log(
                "No text has been extracted yet. Call extract_from_pdf first.",
                level=logging.WARNING,
            )
            return ""

        if page_index < 0 or page_index >= len(self._page_texts):
            raise IndexError(
                f"Page index {page_index} is out of range. PDF has {len(self._page_texts)} pages."
            )

        return self._page_texts[page_index]

    def save_text_to_file(self, output_path: Union[str, Path]) -> None:
        """
        Save the extracted text to a file.

        Args:
            output_path: Path where the text file will be saved

        Raises:
            ValueError: If no text has been extracted yet
        """
        if self._extracted_text is None:
            raise ValueError(
                "No text has been extracted yet. Call extract_from_pdf first."
            )

        # Convert to Path object
        path_obj = Path(output_path) if isinstance(output_path, str) else output_path

        try:
            with open(path_obj, "w", encoding="utf-8") as f:
                f.write(self._extracted_text)

            self.log(f"Text saved to {path_obj}")

        except Exception as e:
            self.log(f"Error saving text to {path_obj}: {str(e)}", level=logging.ERROR)
            raise

    def find_sections(self, section_markers: Dict[str, str]) -> Dict[str, str]:
        """
        Find and extract sections from the PDF text based on section markers.

        Args:
            section_markers: Dictionary mapping section names to marker patterns

        Returns:
            Dictionary mapping section names to extracted text
        """
        if self._extracted_text is None:
            self.log(
                "No text has been extracted yet. Call extract_from_pdf first.",
                level=logging.WARNING,
            )
            return {}

        sections = {}

        for section_name, marker in section_markers.items():
            try:
                pattern = re.compile(
                    marker
                    + r"(.*?)(?=(?:"
                    + "|".join(section_markers.values())
                    + r")|$)",
                    re.IGNORECASE | re.DOTALL,
                )
                matches = pattern.findall(self._extracted_text)

                if matches:
                    sections[section_name] = matches[0].strip()
                    if self.debug:
                        self.log(
                            f"Found section '{section_name}' with {len(sections[section_name])} characters"
                        )
                else:
                    sections[section_name] = ""
                    if self.debug:
                        self.log(f"Section '{section_name}' not found")

            except re.error as e:
                self.log(
                    f"Error in regex pattern for section '{section_name}': {str(e)}",
                    level=logging.ERROR,
                )
                sections[section_name] = ""

        return sections


class DonorPDFExtractor(PDFExtractor):
    """
    Specialized PDF extractor for donor documents.

    This class extends the basic PDFExtractor with methods specific to
    extracting information from donor PDF documents.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, debug: bool = False):
        """
        Initialize the donor PDF extractor.

        Args:
            logger: Optional logger instance
            debug: Enable debug mode for verbose logging
        """
        super().__init__(logger=logger, debug=debug)

        # Define donor document sections
        self.donor_sections = {
            "informations_generales": r"Informations générales",
            "serologies": r"Sérologies",
            "morphologie": r"Morphologie",
            "habitus": r"Habitus",
            "antecedents": r"Antécédents",
            "bilan_infectieux": r"Bilan infectieux",
            "bilan_hemodynamique": r"Bilan hémodynamique",
            "evolution_hemodynamique": r"Evolution hémodynamique",
            # "bilan_pulmonaire": r"Bilan pulmonaire",
            "bilan_cardiaque": r"Bilan cardiaque",
        }

        self.gds_extractor = ImprovedGazDuSangExtractor(logger=logger, debug=debug)
        self.hemo_extractor = ImprovedHemodynamicExtractor(logger=logger, debug=debug)

    def extract_donor_data(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract all donor data from a PDF file with improved extractors.

        Args:
            pdf_path: Path to the donor PDF file

        Returns:
            Dictionary containing the structured donor data
        """
        # Extract text from PDF
        try:
            self.extract_from_pdf(pdf_path)
        except Exception as e:
            self.log(f"Failed to extract text from PDF: {str(e)}", level=logging.ERROR)
            return {"fichier_source": Path(pdf_path).name, "erreur": str(e)}

        # Create result dictionary
        result = {
            "fichier_source": Path(pdf_path).name,
            "chemin_relatif": str(Path(pdf_path)),
            "informations_donneur": self._extract_donor_basic_info(),
        }

        # Extract standard sections using existing patterns
        standard_sections = [
            "serologies",
            "morphologie", 
            "habitus",
            "antecedents",
            "bilan_infectieux",
            "bilan_hemodynamique",
            # "bilan_pulmonaire",
            "bilan_cardiaque_morphologique",
            "thorax"
        ]

        for section_name in standard_sections:
            try:
                patterns = get_pattern_group(section_name)
                result[section_name] = self._extract_section_data(patterns)
            except KeyError:
                self.log(
                    f"Pattern group '{section_name}' not found. Skipping section.",
                    level=logging.WARNING,
                )
                result[section_name] = {}
            except Exception as e:
                self.log(
                    f"Error extracting section '{section_name}': {str(e)}",
                    level=logging.ERROR,
                )
                result[section_name] = {}

        # =========== EXTRACTION AMÉLIORÉE DES PARAMÈTRES RESPIRATOIRES ===========
        # Extract GDS data
        try:
            gds_data = self.gds_extractor.extract_gds_data(pdf_path)
            print(f"DEBUG GDS: {gds_data}")  # Ligne de debug
            result["gds"] = gds_data
        except Exception as e:
            self.log(f"Error extracting GDS data: {str(e)}", level=logging.WARNING)
            result["gds"] = {}

        # ========== EXTRACTION AMÉLIORÉE DE L'ÉVOLUTION HÉMODYNAMIQUE ===========
        try:
            self.log("Extracting hemodynamic evolution using improved extractor")
            hemo_data = self.hemo_extractor.extract_hemodynamic_evolution(self.get_text())
            result["evolution_hemodynamique"] = hemo_data
            
            # Validation physiologique optionnelle
            if hemo_data:
                warnings = self.hemo_extractor.validate_hemodynamic_data(hemo_data)
                if warnings:
                    self.log(
                        f"Hemodynamic validation warnings for {Path(pdf_path).name}: {warnings}", 
                        level=logging.WARNING
                    )
                
                if self.debug:
                    self.log(f"Extracted hemodynamic data: {hemo_data}")
            else:
                self.log("No hemodynamic evolution data extracted", level=logging.WARNING)
                
        except Exception as e:
            self.log(f"Error extracting hemodynamic evolution data: {str(e)}", level=logging.ERROR)
            result["evolution_hemodynamique"] = {}

        # Log final extraction summary
        total_sections = len(standard_sections) + 2  # +2 for GDS and hemodynamic
        extracted_sections = sum(1 for section in result.values() if isinstance(section, dict) and section)
        
        self.log(
            f"Extraction complete for {Path(pdf_path).name}: "
            f"{extracted_sections}/{total_sections} sections successfully extracted"
        )

        return result

    def _extract_donor_basic_info(self) -> Dict[str, Any]:
        """
        Extract basic donor information.

        Returns:
            Dictionary of basic donor information
        """
        result = {}

        # Get donor patterns
        try:
            donor_patterns = get_pattern_group("donor")

            # Extract each field
            for key, pattern in donor_patterns.items():
                try:
                    matches = pattern.findall(self.get_text())
                    if matches:
                        first_match = matches[0]
                        if isinstance(first_match, tuple) and len(first_match) > 0:
                            # We got a tuple from a pattern with groups
                            result[key] = first_match[0].strip()
                        else:
                            # We got a string from a pattern without groups
                            result[key] = str(first_match).strip()
                except Exception as e:
                    self.log(
                        f"Error extracting donor field '{key}': {str(e)}",
                        level=logging.WARNING,
                    )

            # Try alternative patterns for missing fields
            if (
                "type_donneur" not in result
                or not result["type_donneur"]
                or result["type_donneur"] == "::"
            ):
                try:
                    alt_patterns = get_pattern_group("donor_alt")
                    matches = alt_patterns["type_donneur_alt"].findall(self.get_text())
                    if matches:
                        result["type_donneur"] = matches[0].strip()
                except Exception as e:
                    self.log(
                        f"Error extracting alternative donor type: {str(e)}",
                        level=logging.WARNING,
                    )

        except KeyError:
            self.log(
                "Donor patterns not found. Basic info extraction failed.",
                level=logging.ERROR,
            )
        except Exception as e:
            self.log(
                f"Unexpected error in basic info extraction: {str(e)}",
                level=logging.ERROR,
            )

        return result

    def _extract_section_data(self, patterns: Dict[str, re.Pattern]) -> Dict[str, Any]:
        """
        Extract data for a specific section using patterns.

        Args:
            patterns: Dictionary of regex patterns

        Returns:
            Dictionary of extracted values
        """
        result = {}

        # Extract values for each pattern
        for key, pattern in patterns.items():
            try:
                matches = pattern.findall(self.get_text())

                if matches:
                    if self.debug:
                        self.log(f"Matches found for {key}: {matches}")

                    # Handle the match based on its type
                    first_match = matches[0]

                    # For numeric patterns
                    if re.search(r"([0-9.,]+)\s*(mmol|UI|µmol|mg)", pattern.pattern):
                        # Extract numeric values
                        numeric_values = []
                        for match in matches:
                            # Extract value based on whether it's a tuple or string
                            value_str = match[0] if isinstance(match, tuple) else match

                            # Convert to number if possible
                            try:
                                # Replace comma with dot for decimal numbers
                                value = value_str.replace(",", ".")
                                numeric_values.append(float(value))
                            except (ValueError, TypeError):
                                numeric_values.append(value_str)

                        # If single value, use it directly
                        result[key] = (
                            numeric_values
                            if len(numeric_values) > 1
                            else numeric_values[0]
                        )
                    else:
                        # For text values
                        if isinstance(first_match, tuple) and len(first_match) > 0:
                            result[key] = first_match[0].strip()
                        else:
                            result[key] = str(first_match).strip()

            except Exception as e:
                self.log(
                    f"Error extracting pattern '{key}': {str(e)}", level=logging.WARNING
                )

        return result
