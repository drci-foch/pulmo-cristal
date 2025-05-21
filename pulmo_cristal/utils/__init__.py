"""
Utils package for pulmo-cristal.

This package provides utility functions for file operations, logging,
and other common tasks used throughout the package.
"""

from .file_utils import (
    find_pdf_files,
    create_output_path,
    ensure_dir,
    backup_file,
    list_directory_tree,
    get_relative_path,
    batch_process_files,
    create_versioned_filename,
    get_file_info,
    format_file_size,
)

from .logging import (
    setup_logger,
    add_file_handler,
    create_rotating_logger,
    create_timed_rotating_logger,
    log_to_string,
    get_all_loggers,
    set_log_level_for_all,
    create_debug_logger,
    create_audit_logger,
)

__all__ = [
    # File utilities
    "find_pdf_files",
    "create_output_path",
    "ensure_dir",
    "backup_file",
    "list_directory_tree",
    "get_relative_path",
    "batch_process_files",
    "create_versioned_filename",
    "get_file_info",
    "format_file_size",
    # Logging utilities
    "setup_logger",
    "add_file_handler",
    "create_rotating_logger",
    "create_timed_rotating_logger",
    "log_to_string",
    "get_all_loggers",
    "set_log_level_for_all",
    "create_debug_logger",
    "create_audit_logger",
]
