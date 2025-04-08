"""
HLA Data Extractor Module for pulmo-cristal package.

This module handles the extraction of HLA data
from donor PDF documents, using both table extraction via Camelot and
regex-based fallback approaches.
"""

import re
import logging
from typing import Dict, Optional, List, Any, Tuple, Union
from pathlib import Path

# Third-party imports
try:
    import camelot.io as camelot
except ImportError:
    try:
        from camelot import io as camelot
    except ImportError:
        camelot = None

# Local imports
from .base import BaseExtractor


class HLAData(dict):
    """Typed dictionary class for HLA data."""
    pass


class HLAExtractor(BaseExtractor):
    """
    Specialized extractor for HLA data from
    donor PDF documents.
    
    This extractor attempts to find and parse HLA tables in the PDF document.
    It uses Camelot for table extraction and falls back to regex-based
    extraction when table parsing fails.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, debug: bool = False):
        """
        Initialize the HLA extractor.
        
        Args:
            logger: Optional logger instance
            debug: Enable debug mode for verbose logging
        """
        super().__init__(logger=logger)
        self.debug = debug
        
        # Primary regex pattern for HLA data extraction as fallback
        self.hla_basic_pattern = r"A1\s+A2\s+B1\s+B2\s+C1\s+C2\s+DR1\s+DR2[^\n]*\n\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
        
        # Additional patterns for other HLA loci
        self.dqb_pattern = r"DQB\s+DQB\s*\n\s*(\d+)\s+(\d+)"
        self.dp_pattern = r"DP\s+DP\s*\n\s*(\d+)\s+(\d+)"
        
        # Table areas to search for HLA data
        # Format: [left, bottom, right, top] in PDF coordinates
        self.table_areas = [
            "0,100,800,300",    # Primary table area
            "0,0,800,400",      # Expanded area if primary fails
            "0,0,800,800",      # Full page if needed
        ]
        
        # Check if Camelot is available
        if camelot is None:
            self.log("Camelot not available. Will use regex-based extraction only.", 
                     level=logging.WARNING)

    def extract_hla_data(self, pdf_path: str) -> Tuple[Dict[str, str], str]:
        """
        Extract HLA data from the PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple containing:
                - Dictionary of HLA values
                - Status message indicating extraction quality
        """
        self.log(f"Extracting HLA data from {pdf_path}")
        
        # Try table extraction first if Camelot is available
        hla_data = {}
        extraction_status = "FAILED"
        
        if camelot is not None:
            try:
                hla_data = self._extract_hla_with_camelot(pdf_path)
                if hla_data and len(hla_data) >= 6:  # At least A, B, and DR loci
                    extraction_status = "OK" 
                    self.log("Successfully extracted HLA data with Camelot")
                else:
                    self.log("Incomplete HLA data extracted with Camelot", level=logging.WARNING)
                    extraction_status = "INCOMPLETE"
            except Exception as e:
                self.log(f"Error during Camelot extraction: {str(e)}", level=logging.ERROR)
                extraction_status = "ERROR_CAMELOT"
        
        # If Camelot failed or is not available, try regex
        if not hla_data or len(hla_data) < 6:
            self.log("Attempting regex-based HLA extraction")
            
            # We need the text content for regex extraction
            if hasattr(self, 'text_content') and self.text_content:
                text_content = self.text_content
            else:
                # If text wasn't provided, we need to extract it
                try:
                    from PyPDF2 import PdfReader
                    with open(pdf_path, 'rb') as f:
                        reader = PdfReader(f)
                        text_content = ""
                        for page in reader.pages:
                            text_content += page.extract_text() + "\n\n"
                except Exception as e:
                    self.log(f"Error extracting text for regex HLA: {str(e)}", 
                             level=logging.ERROR)
                    text_content = ""
            
            # Try regex extraction
            regex_hla = self._extract_hla_with_regex(text_content)
            
            # Merge with any Camelot results, preferring Camelot data
            for key, value in regex_hla.items():
                if key not in hla_data or not hla_data[key]:
                    hla_data[key] = value
            
            # Update extraction status
            if regex_hla and len(regex_hla) >= 6:
                if extraction_status in ("FAILED", "ERROR_CAMELOT"):
                    extraction_status = "OK_REGEX"
                elif extraction_status == "INCOMPLETE":
                    extraction_status = "OK_MIXED"
            else:
                if extraction_status == "INCOMPLETE":
                    extraction_status = "INCOMPLETE_MIXED"
                elif extraction_status in ("FAILED", "ERROR_CAMELOT"):
                    extraction_status = "INCOMPLETE_REGEX"
        
        # Create the final HLA data with default values where needed
        final_hla = self._create_standardized_hla(hla_data)
        
        # Final status check
        if extraction_status in ("OK", "OK_REGEX", "OK_MIXED") and len(final_hla) >= 8:
            status = extraction_status
        else:
            status = "À VÉRIFIER MANUELLEMENT"
            
        return final_hla, status

    def _extract_hla_with_camelot(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract HLA data using Camelot for table detection.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary of HLA values
        """
        if camelot is None:
            self.log("Camelot is not available", level=logging.WARNING)
            return {}
            
        hla_data = {}
        
        # Try each table area in sequence
        for area in self.table_areas:
            try:
                if self.debug:
                    self.log(f"Trying to extract HLA table with area: {area}")
                    
                tables = camelot.read_pdf(
                    filepath=pdf_path,
                    flavor="stream",  # "stream" works better for text-based tables
                    pages="1",        # Typically HLA is on first page
                    table_areas=[area],
                )
                
                if tables and len(tables) > 0:
                    # Process each table found
                    for i, table in enumerate(tables):
                        if self.debug:
                            self.log(f"Processing table {i+1}/{len(tables)}")
                        
                        # Check if this looks like an HLA table
                        table_hla = self._process_hla_table(table)
                        
                        # If we found HLA data, merge it with existing data
                        if table_hla:
                            hla_data.update(table_hla)
                            
                            # If we have at least the 8 basic HLA values (A,B,C,DR), stop searching
                            if all(key in hla_data for key in 
                                  ["A1", "A2", "B1", "B2", "C1", "C2", "DR1", "DR2"]):
                                return hla_data
            
            except Exception as e:
                self.log(f"Error extracting HLA with area {area}: {str(e)}", 
                        level=logging.WARNING)
                continue
        
        return hla_data

    def _process_hla_table(self, table: Any) -> Dict[str, str]:
        """
        Process a table to extract HLA data.
        
        Args:
            table: Camelot table object
            
        Returns:
            Dictionary of HLA values
        """
        hla_data = {}
        
        try:
            # Convert to DataFrame for easier processing
            df = table.df if hasattr(table, "df") else table
            
            # Look for HLA header row
            header_row = None
            data_row = None
            
            # First pass: Look for exact HLA headers
            for row_idx, row in df.iterrows():
                row_text = " ".join(str(cell) for cell in row)
                row_text = row_text.lower()
                
                # Check if this looks like an HLA header row
                if any(marker in row_text for marker in 
                      ["hla", "a1", "a2", "b1", "b2", "dr1", "dr2"]):
                    header_row = row_idx
                    # Data row is usually the next row
                    if row_idx + 1 < len(df):
                        data_row = row_idx + 1
                    break
            
            # If we found a header and data row
            if header_row is not None and data_row is not None:
                headers = [str(cell).strip() for cell in df.iloc[header_row]]
                data = [str(cell).strip() for cell in df.iloc[data_row]]
                
                # Process the headers and data
                for i, header in enumerate(headers):
                    if i < len(data):
                        # Clean up header and data
                        clean_header = self._clean_hla_header(header)
                        clean_data = self._clean_hla_value(data[i])
                        
                        if clean_header and clean_data:
                            hla_data[clean_header] = clean_data
            
            # Second pass: Try to infer columns if headers aren't clear
            if not hla_data and len(df) >= 2:
                # Assume first row is header, second row is data
                headers = ["A1", "A2", "B1", "B2", "C1", "C2", "DR1", "DR2"]
                data = [str(cell).strip() for cell in df.iloc[1]]
                
                # Only use this if data looks like HLA values (all numeric)
                if all(self._is_valid_hla_value(val) for val in data[:len(headers)]):
                    for i, header in enumerate(headers):
                        if i < len(data):
                            hla_data[header] = self._clean_hla_value(data[i])
        
        except Exception as e:
            self.log(f"Error processing potential HLA table: {str(e)}", 
                    level=logging.WARNING)
        
        return hla_data

    def _extract_hla_with_regex(self, text: str) -> Dict[str, str]:
        """
        Extract HLA data using regex patterns as fallback.
        
        Args:
            text: Text content from the PDF
            
        Returns:
            Dictionary of HLA values
        """
        hla_data = {}
        
        # Extract basic HLA (A, B, C, DR)
        try:
            basic_matches = re.findall(self.hla_basic_pattern, text, re.DOTALL)
            if basic_matches and len(basic_matches) > 0:
                if isinstance(basic_matches[0], tuple) and len(basic_matches[0]) >= 8:
                    values = basic_matches[0]
                    hla_data.update({
                        "A1": self._clean_hla_value(values[0]),
                        "A2": self._clean_hla_value(values[1]),
                        "B1": self._clean_hla_value(values[2]),
                        "B2": self._clean_hla_value(values[3]),
                        "C1": self._clean_hla_value(values[4]),
                        "C2": self._clean_hla_value(values[5]),
                        "DR1": self._clean_hla_value(values[6]),
                        "DR2": self._clean_hla_value(values[7])
                    })
        except re.error as e:
            self.log(f"Error in basic HLA regex: {str(e)}", level=logging.ERROR)
        
        # Extract DQB values
        try:
            dqb_matches = re.findall(self.dqb_pattern, text, re.DOTALL)
            if dqb_matches and len(dqb_matches) > 0:
                if isinstance(dqb_matches[0], tuple) and len(dqb_matches[0]) >= 2:
                    hla_data.update({
                        "DQB1": self._clean_hla_value(dqb_matches[0][0]),
                        "DQB2": self._clean_hla_value(dqb_matches[0][1])
                    })
        except re.error as e:
            self.log(f"Error in DQB regex: {str(e)}", level=logging.ERROR)
        
        # Extract DP values
        try:
            dp_matches = re.findall(self.dp_pattern, text, re.DOTALL)
            if dp_matches and len(dp_matches) > 0:
                if isinstance(dp_matches[0], tuple) and len(dp_matches[0]) >= 2:
                    hla_data.update({
                        "DP1": self._clean_hla_value(dp_matches[0][0]),
                        "DP2": self._clean_hla_value(dp_matches[0][1])
                    })
        except re.error as e:
            self.log(f"Error in DP regex: {str(e)}", level=logging.ERROR)
        
        return hla_data

    def _clean_hla_header(self, header: str) -> str:
        """
        Clean and normalize HLA header values.
        
        Args:
            header: Raw header text
            
        Returns:
            Normalized header name
        """
        # Strip whitespace and convert to uppercase
        header = header.strip().upper()
        
        # Handle complex headers (e.g., "A1\nA2")
        if "\n" in header:
            parts = header.split("\n")
            return parts[0].strip()
            
        # Map common header variations
        header_mapping = {
            'A': 'A1', 'A1': 'A1',
            'A2': 'A2',
            'B': 'B1', 'B1': 'B1',
            'B2': 'B2',
            'C': 'C1', 'C1': 'C1',
            'C2': 'C2',
            'DR': 'DR1', 'DR1': 'DR1',
            'DR2': 'DR2',
            'DQ': 'DQB1', 'DQ1': 'DQB1', 'DQB': 'DQB1', 'DQB1': 'DQB1',
            'DQ2': 'DQB2', 'DQB2': 'DQB2',
            'DP': 'DP1', 'DP1': 'DP1',
            'DP2': 'DP2'
        }
        
        # Try to match with known patterns
        for pattern, mapped_header in header_mapping.items():
            if pattern in header:
                return mapped_header
                
        # If no match, return empty string
        return ""

    def _clean_hla_value(self, value: str) -> str:
        """
        Clean and validate HLA values.
        
        Args:
            value: Raw HLA value
            
        Returns:
            Cleaned HLA value or empty string if invalid
        """
        # Remove whitespace and normalize
        value = value.strip()
        
        # Check if this looks like a valid HLA value
        if self._is_valid_hla_value(value):
            return value
        
        return ""

    def _is_valid_hla_value(self, value: str) -> bool:
        """
        Check if a string looks like a valid HLA value.
        
        Args:
            value: String to validate
            
        Returns:
            True if valid HLA value format
        """
        # Most HLA values are numeric, or numeric with a suffix
        if re.match(r'^\d+[wW]?$', value):
            return True
            
        # Empty strings are not valid
        if not value or value == "--" or value == "NA":
            return False
            
        # Some HLA values may contain a dash
        if re.match(r'^\d+-\d+$', value):
            return True
            
        return False

    def _create_standardized_hla(self, hla_data: Dict[str, str]) -> Dict[str, str]:
        """
        Create a standardized HLA dictionary with all expected keys.
        
        Args:
            hla_data: Extracted HLA data
            
        Returns:
            Standardized HLA dictionary with default values for missing keys
        """
        standard_keys = [
            "A1", "A2", "B1", "B2", "C1", "C2", "DR1", "DR2", 
            "DQB1", "DQB2", "DP1", "DP2"
        ]
        
        standardized = HLAData()
        default_value = "À AJOUTER"
        
        for key in standard_keys:
            if key in hla_data and hla_data[key]:
                standardized[key] = hla_data[key]
            else:
                standardized[key] = default_value
                
        return standardized