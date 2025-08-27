"""
Hemodynamic Evolution Extractor Module for pulmo-cristal package.

This module handles the extraction of hemodynamic evolution data from donor PDF documents,
with proper handling of temporal tables and extraction of the most recent values.
"""

import re
import logging
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime

# Local imports
from .base import BaseExtractor


class ImprovedHemodynamicExtractor(BaseExtractor):
    """
    Specialized extractor for hemodynamic evolution data from donor PDF documents.

    This extractor handles the complex table structure of hemodynamic measurements,
    ensuring that the most recent (rightmost/latest timestamp) values are extracted.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, debug: bool = False):
        """
        Initialize the hemodynamic extractor.

        Args:
            logger: Optional logger instance
            debug: Enable debug mode for verbose logging
        """
        super().__init__(logger=logger)
        self.debug = debug

    def extract_hemodynamic_evolution(self, text: str) -> Dict[str, float]:
        """
        Extract hemodynamic evolution data with improved temporal handling.
        
        Finds the hemodynamic evolution table and extracts the values from
        the rightmost column (most recent timestamp).
        
        Args:
            text: Full text content from the PDF

        Returns:
            Dictionary containing the most recent hemodynamic values
        """
        if not text:
            self.log("No text content provided for hemodynamic extraction", level=logging.WARNING)
            return {}

        data = {}
        
        # Find the hemodynamic evolution table
        table_pattern = r"Evolution hémodynamique.*?(?=\n\s*\n|\Z)"
        table_match = re.search(table_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if not table_match:
            # Try alternative pattern
            table_pattern_alt = r"Evolution hemodynamique.*?(?=\n\s*\n|\Z)"
            table_match = re.search(table_pattern_alt, text, re.DOTALL | re.IGNORECASE)
            
        if not table_match:
            self.log("Hemodynamic evolution table not found", level=logging.WARNING)
            return {}
            
        table_text = table_match.group(0)
        
        if self.debug:
            self.log(f"Found hemodynamic table: {table_text[:300]}...")

        # Extract and parse timestamps to understand the table structure
        timestamps = self._extract_timestamps(table_text)
        
        if timestamps:
            if self.debug:
                self.log(f"Found {len(timestamps)} timestamps: {timestamps}")
            
            # Use structured extraction based on timestamps
            data = self._extract_by_timestamp_columns(table_text, timestamps)
        else:
            self.log("Could not find timestamp structure, using fallback extraction", level=logging.WARNING)
            data = self._fallback_extraction(table_text)

        # Ensure all expected medications are present with default values
        expected_medications = ["dopamine", "dobutamine", "adrenaline", "noradrenaline"]
        for med in expected_medications:
            if med not in data:
                data[med] = 0.0

        if self.debug:
            self.log(f"Final hemodynamic data: {data}")

        return data
    
    def _extract_timestamps(self, table_text: str) -> List[Tuple[str, str]]:
        """
        Extract timestamps from the hemodynamic table.
        
        Args:
            table_text: The hemodynamic table text
            
        Returns:
            List of (date, time) tuples in chronological order
        """
        timestamps = []
        
        # Pattern for date and time pairs
        datetime_pattern = r"(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})"
        matches = re.findall(datetime_pattern, table_text)
        
        if matches:
            timestamps = matches
        else:
            # Try date-only pattern
            date_pattern = r"(\d{2}/\d{2}/\d{4})"
            dates = re.findall(date_pattern, table_text)
            timestamps = [(date, "") for date in dates]
        
        return timestamps
    
    def _extract_by_timestamp_columns(self, table_text: str, timestamps: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Extract medication values using timestamp-based column structure.
        
        Args:
            table_text: The hemodynamic table text
            timestamps: List of timestamp tuples
            
        Returns:
            Dictionary with medication values from the last timestamp
        """
        data = {}
        
        # For each medication, extract the row and parse values by column
        medications = {
            "dopamine": {"pattern": r"dopamine", "unit": "gamma.k/mn"},
            "dobutamine": {"pattern": r"dobutamine", "unit": "gamma.k/mn"}, 
            "adrenaline": {"pattern": r"adrénaline", "unit": "mg/h"},
            "noradrenaline": {"pattern": r"noradrénaline", "unit": "mg/h"}
        }
        
        for med_name, med_info in medications.items():
            values = self._extract_medication_row_values(table_text, med_info["pattern"], med_info["unit"])
            
            if values and len(values) > 0:
                # Take the last value (most recent timestamp)
                data[med_name] = values[-1]
                if self.debug:
                    self.log(f"Extracted {med_name}: {values} -> using latest: {values[-1]}")
            else:
                data[med_name] = 0.0
                if self.debug:
                    self.log(f"No values found for {med_name}, defaulting to 0.0")

        return data
    
    def _extract_medication_row_values(self, table_text: str, med_pattern: str, unit: str) -> List[float]:
        """
        Extract all values for a specific medication from its table row.
        
        Args:
            table_text: The hemodynamic table text
            med_pattern: Regex pattern to match the medication name
            unit: Unit of measurement (gamma.k/mn or mg/h)
            
        Returns:
            List of values in chronological order
        """
        values = []
        
        # Find the line containing this medication
        line_pattern = f"{med_pattern}.*?(?=\\n|$)"
        line_match = re.search(line_pattern, table_text, re.IGNORECASE | re.DOTALL)
        
        if not line_match:
            if self.debug:
                self.log(f"Could not find line for medication: {med_pattern}")
            return values
        
        medication_line = line_match.group(0)
        
        if self.debug:
            self.log(f"Medication line for {med_pattern}: {medication_line}")
        
        # Extract all numeric values from this line based on unit
        if unit == "gamma.k/mn":
            # Pattern: number followed by gamma.k/mn or just number in gamma.k/mn context
            value_patterns = [
                r"(\d+(?:\.\d+)?)\s*gamma\.k/mn",
                r"(\d+(?:\.\d+)?)\s*gamma\.k\/mn",
                r"(\d+(?:\.\d+)?)\s*γ\.k/mn"
            ]
        else:  # mg/h
            # Pattern: number followed by mg/h
            value_patterns = [
                r"(\d+(?:\.\d+)?)\s*mg/h",
                r"(\d+(?:\.\d+)?)\s*mg\/h"
            ]
        
        # Try each pattern to find values
        for pattern in value_patterns:
            matches = re.findall(pattern, medication_line)
            if matches:
                for match in matches:
                    try:
                        values.append(float(match))
                    except ValueError:
                        continue
                break  # Stop after first successful pattern
        
        # If no unit-specific values found, try to extract standalone numbers
        # This handles cases where the unit might be in the header only
        if not values:
            # Look for standalone numbers in typical medication dose ranges
            number_pattern = r"(\d+(?:\.\d+)?)"
            numbers = re.findall(number_pattern, medication_line)
            
            for num_str in numbers:
                try:
                    num = float(num_str)
                    # Apply reasonable range filters based on medication type
                    if unit == "gamma.k/mn" and 0 <= num <= 50:  # Dopamine/Dobutamine range
                        values.append(num)
                    elif unit == "mg/h" and 0 <= num <= 20:  # Adrenaline/Noradrenaline range
                        values.append(num)
                except ValueError:
                    continue
                
        return values
    
    def _fallback_extraction(self, table_text: str) -> Dict[str, float]:
        """
        Fallback extraction method when table structure is unclear.
        
        Args:
            table_text: The hemodynamic table text
            
        Returns:
            Dictionary with extracted values using simple patterns
        """
        data = {}
        
        # Enhanced patterns to extract the last occurrence of each medication
        patterns = {
            "dopamine": [
                r"dopamine[^\n]*?(\d+(?:\.\d+)?)\s*gamma\.k/mn(?:[^\n]*(?:\d+(?:\.\d+)?)\s*gamma\.k/mn)*[^\n]*$",
                r"dopamine.*?(\d+(?:\.\d+)?)[^\d\n]*$"
            ],
            "dobutamine": [
                r"dobutamine[^\n]*?(\d+(?:\.\d+)?)\s*gamma\.k/mn(?:[^\n]*(?:\d+(?:\.\d+)?)\s*gamma\.k/mn)*[^\n]*$",
                r"dobutamine.*?(\d+(?:\.\d+)?)[^\d\n]*$"
            ],
            "adrenaline": [
                r"adrénaline[^\n]*?(\d+(?:\.\d+)?)\s*mg/h(?:[^\n]*(?:\d+(?:\.\d+)?)\s*mg/h)*[^\n]*$",
                r"adrénaline.*?(\d+(?:\.\d+)?)[^\d\n]*$"
            ],
            "noradrenaline": [
                r"noradrénaline[^\n]*?(\d+(?:\.\d+)?)\s*mg/h(?:[^\n]*(?:\d+(?:\.\d+)?)\s*mg/h)*[^\n]*$",
                r"noradrénaline.*?(\d+(?:\.\d+)?)[^\d\n]*$"
            ]
        }
        
        for med_name, pattern_list in patterns.items():
            found = False
            
            # Try each pattern until one works
            for pattern in pattern_list:
                # Find all matches and take the last one
                matches = re.findall(pattern, table_text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    try:
                        # If multiple matches, take the last one
                        last_match = matches[-1] if isinstance(matches[-1], str) else matches[-1][-1]
                        value = float(last_match)
                        data[med_name] = value
                        found = True
                        if self.debug:
                            self.log(f"Fallback extracted {med_name}: {value}")
                        break
                    except (ValueError, IndexError, TypeError):
                        continue
            
            if not found:
                data[med_name] = 0.0
        
        if self.debug:
            self.log(f"Fallback extraction results: {data}")
            
        return data

    def validate_hemodynamic_data(self, data: Dict[str, float]) -> List[str]:
        """
        Validate extracted hemodynamic data for physiological plausibility.
        
        Args:
            data: Dictionary of extracted hemodynamic values
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Dopamine/Dobutamine should be between 0 and 50 gamma/kg/min typically
        for med in ["dopamine", "dobutamine"]:
            if med in data:
                if data[med] < 0:
                    warnings.append(f"{med} value {data[med]} is negative")
                elif data[med] > 50:
                    warnings.append(f"{med} value {data[med]} seems very high (>50 gamma/kg/min)")
        
        # Adrenaline/Noradrenaline should be between 0 and 10 mg/h typically  
        for med in ["adrenaline", "noradrenaline"]:
            if med in data:
                if data[med] < 0:
                    warnings.append(f"{med} value {data[med]} is negative")
                elif data[med] > 10:
                    warnings.append(f"{med} value {data[med]} seems very high (>10 mg/h)")
        
        return warnings

    def extract_debug_info(self, text: str) -> Dict[str, Any]:
        """
        Extract debug information about the hemodynamic table structure.
        
        Args:
            text: Full text content from the PDF
            
        Returns:
            Dictionary with debug information
        """
        debug_info = {}
        
        # Find the hemodynamic evolution table
        table_pattern = r"Evolution hémodynamique.*?(?=\n\s*\n|\Z)"
        table_match = re.search(table_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if table_match:
            table_text = table_match.group(0)
            debug_info["table_found"] = True
            debug_info["table_length"] = len(table_text)
            debug_info["table_preview"] = table_text[:500]
            
            # Extract timestamps
            timestamps = self._extract_timestamps(table_text)
            debug_info["timestamps"] = timestamps
            debug_info["timestamp_count"] = len(timestamps)
            
            # Check for medication lines
            medications = ["dopamine", "dobutamine", "adrénaline", "noradrénaline"]
            debug_info["medications_found"] = {}
            
            for med in medications:
                line_pattern = f"{med}.*?(?=\\n|$)"
                line_match = re.search(line_pattern, table_text, re.IGNORECASE)
                debug_info["medications_found"][med] = bool(line_match)
                if line_match:
                    debug_info[f"{med}_line"] = line_match.group(0)
            
        else:
            debug_info["table_found"] = False
            
        return debug_info