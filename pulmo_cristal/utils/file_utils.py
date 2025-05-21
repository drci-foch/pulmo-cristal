"""
File Utilities Module for pulmo-cristal package.

This module provides utility functions for file operations,
including file discovery, naming, and path handling.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union, Tuple, Generator, Dict, Any
import re
import shutil
import logging


def find_pdf_files(
    directory: Union[str, Path],
    recursive: bool = True,
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
) -> List[Path]:
    """
    Find PDF files in a directory.

    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories
        include_pattern: Regex pattern that filenames must match
        exclude_pattern: Regex pattern that filenames must not match

    Returns:
        List of Path objects to PDF files
    """
    # Convert to Path object
    base_dir = Path(directory)

    # Compile regex patterns if provided
    include_regex = (
        re.compile(include_pattern, re.IGNORECASE) if include_pattern else None
    )
    exclude_regex = (
        re.compile(exclude_pattern, re.IGNORECASE) if exclude_pattern else None
    )

    # Find all PDF files
    if recursive:
        all_pdfs = list(base_dir.glob("**/*.pdf"))
    else:
        all_pdfs = list(base_dir.glob("*.pdf"))

    # Filter files based on patterns if provided
    result = []
    for pdf_path in all_pdfs:
        # Get the filename without the path
        filename = pdf_path.name

        # Check patterns
        include_match = not include_regex or include_regex.search(filename)
        exclude_match = exclude_regex and exclude_regex.search(filename)

        # Add to result if it matches include and doesn't match exclude
        if include_match and not exclude_match:
            result.append(pdf_path)

    return result


def create_output_path(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    extension: str = ".json",
    add_timestamp: bool = True,
    timestamp_format: str = "%Y%m%d_%H%M%S",
) -> Path:
    """
    Create an output path based on an input path.

    Args:
        input_path: Original file path
        output_dir: Directory for the output file
        extension: File extension for the output file
        add_timestamp: Whether to add a timestamp to the filename
        timestamp_format: Format for the timestamp

    Returns:
        Path object for the output file
    """
    # Convert to Path objects
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the stem (filename without extension)
    stem = input_path.stem

    # Add timestamp if requested
    if add_timestamp:
        timestamp = datetime.now().strftime(timestamp_format)
        new_filename = f"{stem}_{timestamp}{extension}"
    else:
        new_filename = f"{stem}{extension}"

    # Create the full output path
    return output_dir / new_filename


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Directory path

    Returns:
        Path object to the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def backup_file(
    file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    Create a backup of a file.

    Args:
        file_path: Path to the file to backup
        backup_dir: Directory for backups (defaults to same directory)

    Returns:
        Path to the backup file
    """
    file_path = Path(file_path)

    # If no backup directory is specified, use the file's directory
    if backup_dir is None:
        backup_dir = file_path.parent
    else:
        backup_dir = Path(backup_dir)
        # Ensure backup directory exists
        backup_dir.mkdir(parents=True, exist_ok=True)

    # Generate a backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_filename

    # Copy the file
    shutil.copy2(file_path, backup_path)
    return backup_path


def list_directory_tree(
    directory: Union[str, Path],
    max_depth: int = 3,
    indent: str = "  ",
    file_types: Optional[List[str]] = None,
) -> str:
    """
    Generate a formatted string representation of a directory tree.

    Args:
        directory: Directory to list
        max_depth: Maximum recursion depth
        indent: Indentation string for each level
        file_types: List of file extensions to include (e.g., ['.pdf', '.txt'])

    Returns:
        Formatted string representation of the directory tree
    """
    directory = Path(directory)
    result = [str(directory) + "/"]
    file_count = 0
    dir_count = 0

    def _list_dir(path: Path, prefix: str, depth: int) -> Tuple[int, int]:
        nonlocal file_count, dir_count

        if depth > max_depth:
            return file_count, dir_count

        # Get items in the directory
        items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))

        # Process each item
        count = len(items)
        for i, item in enumerate(items):
            is_last = i == count - 1

            # Choose the appropriate prefix characters
            if is_last:
                branch = "└── "
                new_prefix = prefix + "    "
            else:
                branch = "├── "
                new_prefix = prefix + "│   "

            if item.is_dir():
                # Directory
                result.append(f"{prefix}{branch}{item.name}/")
                dir_count += 1
                _list_dir(item, new_prefix, depth + 1)
            else:
                # File - check if it matches the file_types filter
                if file_types is None or any(
                    item.name.lower().endswith(ext.lower()) for ext in file_types
                ):
                    result.append(f"{prefix}{branch}{item.name}")
                    file_count += 1

        return file_count, dir_count

    _list_dir(directory, "", 1)
    result.append(f"\nTotal: {dir_count} directories, {file_count} files")

    return "\n".join(result)


def get_relative_path(file_path: Union[str, Path], base_dir: Union[str, Path]) -> str:
    """
    Get the relative path of a file with respect to a base directory.

    Args:
        file_path: Path to the file
        base_dir: Base directory

    Returns:
        Relative path as a string
    """
    file_path = Path(file_path)
    base_dir = Path(base_dir)

    try:
        return str(file_path.relative_to(base_dir))
    except ValueError:
        # If file_path is not relative to base_dir, return the absolute path
        return str(file_path)


def batch_process_files(
    files: List[Union[str, Path]],
    batch_size: int = 10,
    logger: Optional[logging.Logger] = None,
) -> Generator[List[Path], None, None]:
    """
    Process files in batches with progress tracking.

    Args:
        files: List of file paths to process
        batch_size: Number of files to include in each batch
        logger: Optional logger for progress messages

    Yields:
        Batches of Path objects
    """
    total_files = len(files)

    if logger:
        logger.info(f"Starting batch processing of {total_files} files")

    # Convert all to Path objects
    file_paths = [Path(f) for f in files]

    # Process in batches
    for i in range(0, total_files, batch_size):
        batch = file_paths[i : i + batch_size]

        if logger:
            logger.info(
                f"Processing batch {i // batch_size + 1}/{(total_files + batch_size - 1) // batch_size} ({len(batch)} files)"
            )

        yield batch


def create_versioned_filename(
    base_name: str,
    extension: str,
    version: str = "0.1.0",
    add_timestamp: bool = True,
    timestamp_format: str = "%Y%m%d_%H%M%S",
) -> str:
    """
    Create a filename with version and optional timestamp.

    Args:
        base_name: Base name for the file
        extension: File extension
        version: Version string
        add_timestamp: Whether to add a timestamp
        timestamp_format: Format for the timestamp

    Returns:
        Formatted filename
    """
    # Ensure the extension starts with a dot
    if not extension.startswith("."):
        extension = "." + extension

    # Add timestamp if requested
    if add_timestamp:
        timestamp = datetime.now().strftime(timestamp_format)
        return f"{base_name}_v{version}_{timestamp}{extension}"
    else:
        return f"{base_name}_v{version}{extension}"


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file information
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    stats = file_path.stat()

    return {
        "name": file_path.name,
        "extension": file_path.suffix,
        "size_bytes": stats.st_size,
        "size_formatted": format_file_size(stats.st_size),
        "created": datetime.fromtimestamp(stats.st_ctime),
        "modified": datetime.fromtimestamp(stats.st_mtime),
        "accessed": datetime.fromtimestamp(stats.st_atime),
        "is_directory": file_path.is_dir(),
        "absolute_path": str(file_path.absolute()),
        "parent_directory": str(file_path.parent),
    }


def format_file_size(size_bytes: int) -> str:
    """
    Format a file size in bytes to a human-readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        Formatted file size string
    """
    # Define size units
    units = ["B", "KB", "MB", "GB", "TB"]

    # Calculate the appropriate unit
    unit_index = 0
    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024
        unit_index += 1

    # Format the result with appropriate precision
    if unit_index == 0:
        # For bytes, no decimal places
        return f"{size_bytes:.0f} {units[unit_index]}"
    else:
        # For larger units, two decimal places
        return f"{size_bytes:.2f} {units[unit_index]}"
