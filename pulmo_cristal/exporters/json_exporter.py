"""
JSON Exporter Module for pulmo-cristal package.

This module provides functionality to export donor data to JSON format,
with support for different encoding options, pretty printing,
and validation features.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Local imports
try:
    from ..extractors.base import BaseExtractor
except ImportError:
    # If used standalone outside package structure
    BaseExtractor = object


class DonorJSONExporter(BaseExtractor):
    """
    Exporter for generating JSON files from donor data.

    This class handles the conversion of donor data dictionaries into JSON format,
    with support for different encoding options, pretty printing, and validation.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        encoding: str = "utf-8",
        indent: int = 4,
        ensure_ascii: bool = False,
    ):
        """
        Initialize the JSON exporter.

        Args:
            logger: Optional logger instance
            encoding: File encoding (default: utf-8)
            indent: JSON indentation level (default: 4)
            ensure_ascii: Whether to escape non-ASCII characters (default: False)
        """
        super().__init__(logger=logger)
        self.encoding = encoding
        self.indent = indent
        self.ensure_ascii = ensure_ascii

    def export_json(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        output_path: Union[str, Path],
        add_timestamp: bool = True,
        add_metadata: bool = True,
    ) -> str:
        """
        Generate a JSON file from donor data.

        Args:
            data: Dictionary or list of dictionaries containing donor data
            output_path: Path where the JSON file will be saved
            add_timestamp: Whether to add a timestamp to the filename
            add_metadata: Whether to add export metadata to the JSON

        Returns:
            Path to the generated JSON file
        """
        # Prepare output path with timestamp if requested
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path_obj = Path(output_path)
            filename = f"{path_obj.stem}_{timestamp}{path_obj.suffix}"
            output_path = path_obj.parent / filename
        else:
            output_path = Path(output_path)

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata if requested
        final_data = data
        if add_metadata:
            if isinstance(data, list):
                # For list data, wrap it in a metadata object
                final_data = {
                    "metadata": self._generate_metadata(len(data)),
                    "donors": data,
                }
            else:
                # For single data object, add metadata as a field
                if isinstance(data, dict):
                    final_data = data.copy()
                    final_data["metadata"] = self._generate_metadata(1)

        # Write the JSON file
        try:
            with open(output_path, "w", encoding=self.encoding) as jsonfile:
                json.dump(
                    final_data,
                    jsonfile,
                    ensure_ascii=self.ensure_ascii,
                    indent=self.indent,
                )

            self.log(f"JSON file generated successfully: {output_path}")
            return str(output_path)

        except Exception as e:
            self.log(f"Error generating JSON file: {e}", level=logging.ERROR)
            raise

    def export_json_stream(
        self,
        data_list: List[Dict[str, Any]],
        output_path: Union[str, Path],
        batch_size: int = 10,
    ) -> str:
        """
        Generate a JSON file by streaming data in batches to handle large datasets.

        Args:
            data_list: List of donor data dictionaries
            output_path: Path where the JSON file will be saved
            batch_size: Number of items to process at once

        Returns:
            Path to the generated JSON file
        """
        output_path = Path(output_path)

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w", encoding=self.encoding) as jsonfile:
                # Write the opening metadata and array
                metadata = self._generate_metadata(len(data_list))
                jsonfile.write("{\n")
                jsonfile.write(
                    f'    "metadata": {json.dumps(metadata, ensure_ascii=self.ensure_ascii, indent=self.indent)},\n'
                )
                jsonfile.write('    "donors": [\n')

                # Process data in batches
                for i, donor in enumerate(data_list):
                    # Convert to JSON with proper indentation
                    donor_json = json.dumps(
                        donor, ensure_ascii=self.ensure_ascii, indent=self.indent
                    )

                    # Add indentation to each line to match the file structure
                    indented_lines = [
                        f"        {line}" for line in donor_json.splitlines()
                    ]
                    donor_text = "\n".join(indented_lines)

                    # Add comma if not the last item
                    if i < len(data_list) - 1:
                        donor_text += ","

                    # Write to file
                    jsonfile.write(donor_text + "\n")

                    # Log progress
                    if (i + 1) % batch_size == 0 or i == len(data_list) - 1:
                        self.log(f"Processed {i + 1}/{len(data_list)} donors")

                # Close the JSON structure
                jsonfile.write("    ]\n")
                jsonfile.write("}\n")

            self.log(f"JSON file generated successfully: {output_path}")
            return str(output_path)

        except Exception as e:
            self.log(f"Error generating JSON file: {e}", level=logging.ERROR)
            raise

    def _generate_metadata(self, count: int) -> Dict[str, Any]:
        """
        Generate metadata for the JSON export.

        Args:
            count: Number of donor records

        Returns:
            Dictionary containing metadata
        """
        return {
            "generator": "pulmo-cristal",
            "version": "0.1.0",  # Should come from package version
            "export_date": datetime.now().isoformat(),
            "record_count": count,
            "encoding": self.encoding,
        }

    def validate_json(
        self, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Validate JSON data against a schema.

        Args:
            data: JSON data to validate
            schema: JSON schema to validate against (optional)

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Basic validation without schema
        if not schema:
            # Check for required donor fields
            required_fields = ["informations_donneur", "date_extraction"]
            for field in required_fields:
                if field not in data:
                    errors.append(f"Missing required field: {field}")

            # Check donor info fields
            donor_info = data.get("informations_donneur", {})
            required_donor_fields = ["num_cristal", "type_donneur"]
            for field in required_donor_fields:
                if field not in donor_info or not donor_info[field]:
                    errors.append(f"Missing required donor field: {field}")

        # Schema validation (would use jsonschema library in a full implementation)
        else:
            errors.append("Schema validation not implemented")

        return errors


def generate_json_filename(
    base_name: str = "donneurs_data",
    include_version: bool = True,
    version: str = "0.1.0",
) -> str:
    """
    Generate a standardized JSON filename with timestamp and optional version.

    Args:
        base_name: Base name for the file
        include_version: Whether to include version in filename
        version: Version string to include

    Returns:
        Generated filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if include_version:
        return f"{base_name}_{timestamp}_v{version}.json"
    else:
        return f"{base_name}_{timestamp}.json"
