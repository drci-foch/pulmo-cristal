"""
Base Extractor Module for pulmo-cristal package.

This module provides the base infrastructure for all extractors in the
pulmo-cristal package, including logging, validation, and common utilities.
"""

import logging
from typing import Optional, Any, Dict, List, Tuple, Union
from datetime import datetime


class BaseExtractor:
    """
    Base class for all extractors in the pulmo-cristal package.

    This class provides common functionality such as logging, validation
    utilities, and error handling that can be used by all specialized extractor
    classes.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the base extractor.

        Args:
            logger: Optional logger instance to use. If not provided,
                   a default logger will be created.
        """
        # Set up logging
        self.logger = logger or self._create_default_logger()

    def log(self, message: str, level: int = logging.INFO) -> None:
        """
        Log a message at the specified level.

        Args:
            message: The message to log
            level: The logging level (e.g., logging.INFO, logging.ERROR)
        """
        if self.logger:
            self.logger.log(level, message)

    def _create_default_logger(self) -> logging.Logger:
        """
        Create a default logger for the extractor.

        Returns:
            A configured logger instance
        """
        logger = logging.getLogger(f"{self.__class__.__name__}")

        # Don't add handlers if they already exist
        if not logger.handlers:
            # Configure logger
            logger.setLevel(logging.INFO)

            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)

            # Add the handler to the logger
            logger.addHandler(console_handler)

        return logger

    def validate_data(
        self, data: Dict[str, Any], required_fields: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate extracted data against required fields.

        Args:
            data: Dictionary of extracted data
            required_fields: List of field names that should be present and non-empty

        Returns:
            Tuple containing:
                - Boolean indicating if the data is valid
                - List of validation issues
        """
        issues = []

        for field in required_fields:
            if field not in data:
                issues.append(f"Missing required field: {field}")
            elif not data[field]:
                issues.append(f"Empty required field: {field}")

        return len(issues) == 0, issues

    def validate_date(self, date_string: str, format_string: str = "%d/%m/%Y") -> bool:
        """
        Validate a date string against the expected format.

        Args:
            date_string: Date string to validate
            format_string: Expected date format string (default: %d/%m/%Y)

        Returns:
            True if the date is valid, False otherwise
        """
        try:
            datetime.strptime(date_string, format_string)
            return True
        except ValueError:
            return False

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text string
        """
        # Remove excessive whitespace
        cleaned = " ".join(text.split())

        # Replace common OCR errors if needed
        replacements = {
            "\u2019": "'",  # Right single quotation mark
            "\u2018": "'",  # Left single quotation mark
            "\u201c": '"',  # Left double quotation mark
            "\u201d": '"',  # Right double quotation mark
            "\u2013": "-",  # En dash
            "\u2014": "-",  # Em dash
            "\u00a0": " ",  # Non-breaking space
        }

        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        return cleaned.strip()

    def get_timestamp(self, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Get a formatted timestamp for the current time.

        Args:
            format_string: Format for the timestamp (default: %Y-%m-%d %H:%M:%S)

        Returns:
            Formatted timestamp string
        """
        return datetime.now().strftime(format_string)

    def convert_to_numeric(self, value: str) -> Union[int, float, str]:
        """
        Attempt to convert a string to a numeric value.

        Args:
            value: String value to convert

        Returns:
            Integer, float, or original string if conversion fails
        """
        # Remove any thousand separators and convert comma decimal separator
        cleaned_value = value.replace(" ", "").replace(",", ".")

        # Try to convert to int or float
        try:
            # First try integer
            int_value = int(cleaned_value)
            return int_value
        except ValueError:
            try:
                # Then try float
                float_value = float(cleaned_value)
                # Convert to int if it's a whole number
                if float_value.is_integer():
                    return int(float_value)
                return float_value
            except ValueError:
                # Return original if both conversions fail
                return value

    def get_latest_value(self, values: Union[List[Any], Any]) -> Any:
        """
        Get the latest value from a list or return the value directly if not a list.

        Args:
            values: List of values or single value

        Returns:
            The last value in the list or the original value if not a list
        """
        if isinstance(values, list) and values:
            return values[-1]
        return values
