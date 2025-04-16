"""
HLA Data Extractor Module for pulmo-cristal package.

This module handles the extraction of HLA (Human Leukocyte Antigen) data
from donor PDF documents, using both table extraction via Camelot and
regex-based fallback approaches.
"""

import re
import logging
import sys
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
        print("Warning: camelot-py not installed. HLA extraction will use regex only.")
        # Don't exit - just continue with limited functionality
# Local imports
from .base import BaseExtractor


class HLAData(dict):
    """Typed dictionary class for HLA data."""
    pass


class HLAExtractor(BaseExtractor):
    """
    Specialized extractor for HLA (Human Leukocyte Antigen) data from
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
        self.text_content = None
        
        # Primary regex pattern for HLA data extraction as fallback
        self.hla_basic_pattern = r"A1\s+A2\s+B1\s+B2\s+C1\s+C2\s+DR1\s+DR2[^\n]*\n\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
        
        # Additional patterns for other HLA loci
        self.dqb_pattern = r"DQB\s+DQB\s*\n\s*(\d+)\s+(\d+)"
        self.dp_pattern = r"DP\s+DP\s*\n\s*(\d+)\s+(\d+)"
        
        # Table areas to search for HLA data
        # Format: [left, bottom, right, top] in PDF coordinates
        self.table_areas = [
            "0,100,800,300",    # Primary table area (as used in original code)
            "0,0,800,400",      # Expanded area if primary fails
            "0,0,800,800",      # Full page if needed
        ]
        
        # Check if Camelot is available
        if camelot is None:
            self.log("Camelot not available. Will use regex-based extraction only.", 
                     level=logging.WARNING)

    def extract_hla_data(self, pdf_path: str) -> Tuple[Dict[str, str], str]:
        """
        Extract HLA data from the PDF document directly using the original working approach.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple containing:
                - Dictionary of HLA values
                - Status message indicating extraction quality
        """
        self.log(f"Extracting HLA data from {pdf_path} using original approach")
        
        # Use the exact technique from the original working code
        try:
            # Extract text from PDF for regex fallback
            from PyPDF2 import PdfReader
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                self.text_content = ""
                for page in reader.pages:
                    self.text_content += page.extract_text() + "\n\n"
        
            # Extract tables using Camelot with the exact original parameters
            tables = camelot.read_pdf(
                filepath=pdf_path,
                flavor="stream",
                pages="1",
                table_areas=["0,100,800,300"],
            )
            
            # Process the tables to extract HLA data
            hla_data = {}
            if tables and len(tables) > 0:
                # Use the original extraction logic from DonneurDataExtractor
                hla_data = self._extract_hla_values_from_dataframe_original(tables)
            
            # If Camelot extraction was insufficient, try regex
            if not hla_data or len(hla_data.keys()) < 5:
                self.log("Camelot extraction insufficient, using regex fallback")
                regex_hla = self._extract_hla_with_regex(self.text_content)
                
                # Merge regex results with any Camelot results
                for key, value in regex_hla.items():
                    if key not in hla_data or not hla_data[key]:
                        hla_data[key] = value
            
            # Determine status based on data completeness
            if hla_data and len(hla_data.keys()) >= 5:
                status = "OK"
            else:
                status = "À VÉRIFIER MANUELLEMENT"
                
            # Standardize the HLA data for output
            final_hla = self._create_standardized_hla(hla_data)
            return final_hla, status
                
        except Exception as e:
            self.log(f"Error in HLA extraction: {str(e)}", level=logging.ERROR)
            # Return default values on error
            default_hla = self._create_standardized_hla({})
            return default_hla, "À VÉRIFIER MANUELLEMENT"
        
        

    def _extract_hla_values_from_dataframe_original(self, tables) -> Dict[str, str]:
        """
        Exact implementation from the original working code.
        
        Args:
            tables: List of tables extracted by Camelot
            
        Returns:
            Dictionary of HLA values
        """
        hla_data = {}
        hla_found = False

        for i, table in enumerate(tables):
            # Obtain the DataFrame from the table
            if hasattr(table, "df"):
                # Camelot Table objects have a df attribute
                df = table.df
            else:
                # For other table types
                df = table

            try:
                # Check if "HLA" is in the headers
                header_row = None
                data_row = None

                # Traverse rows to find HLA header and data
                for row_idx, row in df.iterrows():
                    row_text = " ".join(str(cell) for cell in row)

                    if "HLA" in row_text and header_row is None:
                        header_row = row_idx
                        continue

                    # If we found the header, the next row contains the data
                    if header_row is not None and data_row is None:
                        data_row = row_idx
                        break

                # If we found both header and data rows
                if header_row is not None and data_row is not None:
                    # Extract HLA values
                    header_values = [str(cell).strip() for cell in df.iloc[header_row]]
                    data_values = [str(cell).strip() for cell in df.iloc[data_row]]

                    # Print values for debugging
                    if self.debug:
                        self.log(f"Headers: {header_values}")
                        self.log(f"Values: {data_values}")

                    # Create dictionary with HLA values
                    for j, header in enumerate(header_values):
                        # Ignore empty or irrelevant cells
                        if j < len(data_values) and header and data_values[j]:
                            # Process composite headers (like A1\nA2)
                            if "\n" in header:
                                sub_headers = header.split("\n")
                                for k, sub_header in enumerate(sub_headers):
                                    if k < len(data_values[j].split("\n")):
                                        sub_value = data_values[j].split("\n")[k]
                                        hla_data[sub_header.strip()] = sub_value.strip()
                                    else:
                                        hla_data[sub_header.strip()] = ""
                            else:
                                hla_data[header] = data_values[j]

                    hla_found = True
                    break
            except Exception as e:
                self.log(f"Error analyzing table {i}: {str(e)}")
                continue

        # If no HLA value was found, try a different approach
        if not hla_found:
            self.log("Attempting alternative extraction method")
            for i, table in enumerate(tables):
                if hasattr(table, "df"):
                    df = table.df
                else:
                    df = table

                # Search for "HLA" in all cells
                for row_idx, row in df.iterrows():
                    for col_idx, cell in enumerate(row):
                        if isinstance(cell, str) and "HLA" in cell:
                            # Find HLA alleles in subsequent rows
                            if row_idx + 1 < len(df):
                                # Assume next row contains HLA data
                                hla_row = df.iloc[row_idx + 1]

                                # Try to extract basic HLA alleles
                                try:
                                    col_names = df.columns.tolist()
                                    for j, col_name in enumerate(col_names):
                                        if j < len(hla_row):
                                            hla_data[f"col_{j}"] = str(hla_row[j]).strip()
                                except Exception as e:
                                    self.log(f"Error in alternative extraction: {str(e)}")

                            hla_found = True
                            break

                    if hla_found:
                        break
        
        # Print HLA data for debugging
        if self.debug:
            self.log(f"Final HLA data: {hla_data}")
        
        return hla_data


    def _extract_hla_with_regex(self, text: str) -> Dict[str, str]:
        """
        Extract HLA data using regex patterns as fallback.
        
        Args:
            text: Text content from the PDF
            
        Returns:
            Dictionary of HLA values
        """
        if not text:
            self.log("No text content available for regex extraction", level=logging.WARNING)
            return {}
            
        hla_data = {}
        
        # Extract basic HLA (A, B, C, DR)
        try:
            self.log("Applying basic HLA regex pattern")
            basic_matches = re.findall(self.hla_basic_pattern, text, re.DOTALL)
            if basic_matches and len(basic_matches) > 0:
                self.log(f"Found basic HLA matches: {basic_matches}")
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
            self.log("Applying DQB regex pattern")
            dqb_matches = re.findall(self.dqb_pattern, text, re.DOTALL)
            if dqb_matches and len(dqb_matches) > 0:
                self.log(f"Found DQB matches: {dqb_matches}")
                if isinstance(dqb_matches[0], tuple) and len(dqb_matches[0]) >= 2:
                    hla_data.update({
                        "DQB1": self._clean_hla_value(dqb_matches[0][0]),
                        "DQB2": self._clean_hla_value(dqb_matches[0][1])
                    })
        except re.error as e:
            self.log(f"Error in DQB regex: {str(e)}", level=logging.ERROR)
        
        # Extract DP values
        try:
            self.log("Applying DP regex pattern")
            dp_matches = re.findall(self.dp_pattern, text, re.DOTALL)
            if dp_matches and len(dp_matches) > 0:
                self.log(f"Found DP matches: {dp_matches}")
                if isinstance(dp_matches[0], tuple) and len(dp_matches[0]) >= 2:
                    hla_data.update({
                        "DP1": self._clean_hla_value(dp_matches[0][0]),
                        "DP2": self._clean_hla_value(dp_matches[0][1])
                    })
        except re.error as e:
            self.log(f"Error in DP regex: {str(e)}", level=logging.ERROR)
        
        # Try additional simpler patterns if main patterns didn't work
        if not hla_data or len(hla_data) < 3:
            self.log("Trying simpler patterns")
            try:
                # Look for individual HLA values with simpler patterns
                simple_patterns = {
                    "A1": r"A1\s*[:=]?\s*(\d+)",
                    "A2": r"A2\s*[:=]?\s*(\d+)",
                    "B1": r"B1\s*[:=]?\s*(\d+)",
                    "B2": r"B2\s*[:=]?\s*(\d+)",
                    "DR1": r"DR1\s*[:=]?\s*(\d+)",
                    "DR2": r"DR2\s*[:=]?\s*(\d+)"
                }
                
                for key, pattern in simple_patterns.items():
                    if key not in hla_data or not hla_data[key]:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        if matches and matches[0]:
                            hla_data[key] = self._clean_hla_value(matches[0])
            except Exception as e:
                self.log(f"Error in simple patterns: {str(e)}", level=logging.WARNING)
        
        if self.debug:
            self.log(f"HLA data from regex: {hla_data}")
            
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
            'A1': 'A1',
            'A2': 'A2',
            'B1': 'B1',
            'B2': 'B2',
            'C1': 'C1',
            'C2': 'C2',
            'DR1': 'DR1',
            'DR2': 'DR2',
            'DQA':'DQA',
            'DQ1': 'DQB1', 'DQB1': 'DQB1',
            'DQ2': 'DQB2', 'DQB2': 'DQB2',
            'DP1': 'DP1',
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
        value = str(value).strip()
        
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
            "DQA", "DQB", "DP1", "DP2"
        ]
        
        standardized = HLAData()
        default_value = "À AJOUTER"
        
        for key in standard_keys:
            if key in hla_data and hla_data[key]:
                standardized[key] = hla_data[key]
            else:
                standardized[key] = default_value
                
        return standardized