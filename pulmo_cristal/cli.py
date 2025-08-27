"""
Command-line Interface for pulmo-cristal package.

This module provides a command-line interface for extracting data from
donor PDF documents and exporting it to various formats.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import logging
import json
from datetime import datetime
import time

# Package version - should be synced with __init__.py
__version__ = "0.1.0"

# Try to import package modules, handling case where package is not installed
try:
    from pulmo_cristal.extractors import DonorPDFExtractor, HLAExtractor, ImprovedGazDuSangExtractor
    from pulmo_cristal.exporters import DonorCSVExporter, DonorJSONExporter
    from pulmo_cristal.models import Donneur
    from pulmo_cristal.utils import (
        create_rotating_logger,
        find_pdf_files,
        batch_process_files,
        create_versioned_filename,
        list_directory_tree,
    )
except ImportError:
    # For development or direct script usage
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from extractors import DonorPDFExtractor, HLAExtractor
    from exporters import DonorCSVExporter, DonorJSONExporter
    from models import Donneur
    from utils import (
        create_rotating_logger,
        find_pdf_files,
        batch_process_files,
        create_versioned_filename,
        list_directory_tree,
    )


def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up the argument parser for the command-line interface.

    Returns:
        Configured argparse parser
    """
    parser = argparse.ArgumentParser(
        description="Extract data from donor PDF documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Common options
    parser.add_argument(
        "--version", "-V", action="version", version=f"pulmo-cristal {__version__}"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)",
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract", help="Extract data from PDF files"
    )

    extract_parser.add_argument(
        "--input", "-i", required=True, help="Input directory containing PDF files"
    )

    extract_parser.add_argument(
        "--output", "-o", default="./output", help="Output directory for extracted data"
    )

    extract_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "csv", "both"],
        default="both",
        help="Output format",
    )

    extract_parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively process subdirectories",
    )

    extract_parser.add_argument(
        "--pattern", "-p", default=None, help="Regex pattern to filter input files"
    )

    extract_parser.add_argument(
        "--workers", "-w", type=int, default=1, help="Number of worker processes"
    )

    extract_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="Batch size for processing files",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List PDF files in a directory")

    list_parser.add_argument("--input", "-i", required=True, help="Directory to list")

    list_parser.add_argument(
        "--recursive", "-r", action="store_true", help="Recursively list subdirectories"
    )

    list_parser.add_argument(
        "--pattern", "-p", default=None, help="Regex pattern to filter listed files"
    )

    list_parser.add_argument(
        "--depth", "-d", type=int, default=3, help="Maximum depth for directory listing"
    )

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert between output formats"
    )

    convert_parser.add_argument("--input", "-i", required=True, help="Input JSON file")

    convert_parser.add_argument("--output", "-o", required=True, help="Output CSV file")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate extracted data")

    validate_parser.add_argument(
        "--input", "-i", required=True, help="Input JSON file to validate"
    )

    validate_parser.add_argument(
        "--strict", "-s", action="store_true", help="Use strict validation"
    )

    validate_parser.add_argument(
        "--report", "-r", default=None, help="Output validation report file"
    )

    return parser


def configure_logging(verbose: int) -> logging.Logger:
    """
    Configure logging based on verbosity level.

    Args:
        verbose: Verbosity level (0=ERROR, 1=WARNING, 2=INFO, 3+=DEBUG)

    Returns:
        Configured logger
    """
    # Map verbosity to log level
    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(verbose, 3)]

    # Create logs directory
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)

    # Create rotating logger
    logger = create_rotating_logger(
        "pulmo_cristal", logs_dir, level=level, console_output=True
    )

    return logger


def extract_command(args: argparse.Namespace, logger: logging.Logger) -> int:
    """
    Execute the extract command.

    Args:
        args: Command line arguments
        logger: Logger instance

    Returns:
        Exit code (0 for success)
    """
    # Start timer
    start_time = time.time()

    # Find PDF files
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1

    logger.info(f"Searching for PDF files in {input_dir}")
    pdf_files = find_pdf_files(
        input_dir, recursive=args.recursive, include_pattern=args.pattern
    )

    # Check if any PDF files were found
    if not pdf_files:
        logger.error(f"No PDF files found in {input_dir}")
        return 1

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create extractors
    donor_extractor = DonorPDFExtractor(logger=logger, debug=(args.verbose >= 3))
    hla_extractor = HLAExtractor(logger=logger, debug=(args.verbose >= 3))
    gds_extractor = ImprovedGazDuSangExtractor(logger=logger, debug=(args.verbose >= 3))

    # Create output filename
    json_filename = create_versioned_filename(
        "donneurs_data", "json", __version__, True
    )
    csv_filename = create_versioned_filename("donneurs_data", "csv", __version__, True)

    # Create exporters if needed
    json_exporter = None
    csv_exporter = None

    if args.format in ["json", "both"]:
        json_exporter = DonorJSONExporter(logger=logger)
        logger.info(f"Will export to JSON: {output_dir / json_filename}")

    if args.format in ["csv", "both"]:
        csv_exporter = DonorCSVExporter(logger=logger)
        logger.info(f"Will export to CSV: {output_dir / csv_filename}")

    # Process files
    all_data = []

    # Use batch processing for better memory management
    total_success = 0
    total_failures = 0

    logger.info(f"Processing with batch size: {args.batch_size}")

    for batch in batch_process_files(
        pdf_files, batch_size=args.batch_size, logger=logger
    ):
        batch_data = []

        for pdf_file in batch:
            try:
                # Extract donor data
                donor_data = donor_extractor.extract_donor_data(pdf_file)

                # Extract HLA data
                hla_data, status = hla_extractor.extract_hla_data(pdf_file)
                gds_data = gds_extractor._select_best_data_smart(pdf_file)


                # Add HLA data to donor data
                donor_data["informations_donneur"]["hla"] = hla_data
                donor_data["gds"] = gds_data
                donor_data["informations_donneur"]["hla_extraction_status"] = status

                # Add relative path
                donor_data["chemin_relatif"] = str(pdf_file.relative_to(input_dir))

                # Add timestamp if not present
                if "date_extraction" not in donor_data:
                    donor_data["date_extraction"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                # Add to batch data
                batch_data.append(donor_data)
                total_success += 1

                logger.info(f"Successfully processed: {pdf_file.name}")

            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                total_failures += 1

        # Add batch to all data
        all_data.extend(batch_data)

        # Intermediate save for JSON
        if json_exporter and batch_data:
            try:
                temp_json_path = output_dir / f"temp_{json_filename}"
                json_exporter.export_json(all_data, temp_json_path, add_timestamp=False)
                logger.debug(f"Saved intermediate JSON to {temp_json_path}")
            except Exception as e:
                logger.warning(f"Failed to save intermediate JSON: {str(e)}")

    # Final export
    if json_exporter and all_data:
        try:
            json_path = output_dir / json_filename
            json_exporter.export_json(all_data, json_path, add_timestamp=False)
            logger.info(f"Saved final JSON to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save final JSON: {str(e)}")
            return 1

    if csv_exporter and all_data:
        try:
            csv_path = output_dir / csv_filename
            csv_exporter.export_csv(all_data, csv_path, add_timestamp=False)
            logger.info(f"Saved CSV to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {str(e)}")
            return 1

    # Compute summary statistics
    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info(
        f"Processing complete: {total_success} successes, {total_failures} failures"
    )
    logger.info(
        f"Total time: {elapsed_time:.2f} seconds ({elapsed_time / len(pdf_files):.2f} seconds per file)"
    )

    # Return success if at least one file was processed successfully
    return 0 if total_success > 0 else 1


def list_command(args: argparse.Namespace, logger: logging.Logger) -> int:
    """
    Execute the list command.

    Args:
        args: Command line arguments
        logger: Logger instance

    Returns:
        Exit code (0 for success)
    """
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Directory does not exist: {input_dir}")
        return 1

    # List PDF files
    pdf_files = find_pdf_files(
        input_dir, recursive=args.recursive, include_pattern=args.pattern
    )

    if not pdf_files:
        logger.info(f"No PDF files found in {input_dir}")
        return 0

    logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")

    # Print directory tree
    tree = list_directory_tree(input_dir, max_depth=args.depth, file_types=[".pdf"])

    print(tree)

    return 0


def convert_command(args: argparse.Namespace, logger: logging.Logger) -> int:
    """
    Execute the convert command.

    Args:
        args: Command line arguments
        logger: Logger instance

    Returns:
        Exit code (0 for success)
    """
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        return 1

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {input_file} to {output_file}")

    # Load JSON data
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, dict) and "donors" in data:
            donor_data = data["donors"]
        elif isinstance(data, list):
            donor_data = data
        else:
            donor_data = [data]

        logger.info(f"Loaded {len(donor_data)} donor records from {input_file}")

    except Exception as e:
        logger.error(f"Failed to load JSON data: {str(e)}")
        return 1

    # Export to CSV
    try:
        csv_exporter = DonorCSVExporter(logger=logger)
        csv_exporter.export_csv(donor_data, output_file, add_timestamp=False)
        logger.info(f"Successfully converted to CSV: {output_file}")
        return 0

    except Exception as e:
        logger.error(f"Failed to convert to CSV: {str(e)}")
        return 1


def validate_command(args: argparse.Namespace, logger: logging.Logger) -> int:
    """
    Execute the validate command.

    Args:
        args: Command line arguments
        logger: Logger instance

    Returns:
        Exit code (0 for success)
    """
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        return 1

    logger.info(f"Validating {input_file}")

    # Load JSON data
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, dict) and "donors" in data:
            donor_data = data["donors"]
        elif isinstance(data, list):
            donor_data = data
        else:
            donor_data = [data]

        logger.info(f"Loaded {len(donor_data)} donor records from {input_file}")

    except Exception as e:
        logger.error(f"Failed to load JSON data: {str(e)}")
        return 1

    # Validate data
    validation_results = []
    valid_count = 0
    invalid_count = 0

    for i, donor_dict in enumerate(donor_data):
        try:
            # Convert to Donneur object (validates during conversion)
            donneur = Donneur.from_dict(donor_dict)

            # Check if valid
            is_valid = len(donneur.validation_errors) == 0

            # Record result
            result = {
                "id": donneur.id or f"Record_{i + 1}",
                "is_valid": is_valid,
                "errors": donneur.validation_errors,
            }

            validation_results.append(result)

            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                logger.warning(
                    f"Validation errors in {result['id']}: {donneur.validation_errors}"
                )

        except Exception as e:
            logger.error(f"Error validating record {i + 1}: {str(e)}")
            validation_results.append(
                {
                    "id": f"Record_{i + 1}",
                    "is_valid": False,
                    "errors": [f"Exception: {str(e)}"],
                }
            )
            invalid_count += 1

    # Print summary
    logger.info(f"Validation complete: {valid_count} valid, {invalid_count} invalid")

    # Save report if requested
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "summary": {
                            "total": len(donor_data),
                            "valid": valid_count,
                            "invalid": invalid_count,
                            "timestamp": datetime.now().isoformat(),
                        },
                        "results": validation_results,
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Validation report saved to {report_path}")

        except Exception as e:
            logger.error(f"Failed to save validation report: {str(e)}")

    # Return success if all records are valid or if not using strict mode
    if args.strict:
        return 0 if invalid_count == 0 else 1
    else:
        return 0


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the command-line interface.

    Args:
        args: Command-line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success)
    """
    # Parse arguments
    parser = setup_argparse()
    parsed_args = parser.parse_args(args)

    # If no command specified, show help
    if not parsed_args.command:
        parser.print_help()
        return 1

    # Configure logging
    logger = configure_logging(parsed_args.verbose)
    logger.debug(f"Arguments: {parsed_args}")

    try:
        # Dispatch to appropriate command
        if parsed_args.command == "extract":
            return extract_command(parsed_args, logger)
        elif parsed_args.command == "list":
            return list_command(parsed_args, logger)
        elif parsed_args.command == "convert":
            return convert_command(parsed_args, logger)
        elif parsed_args.command == "validate":
            return validate_command(parsed_args, logger)
        else:
            logger.error(f"Unknown command: {parsed_args.command}")
            return 1

    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
