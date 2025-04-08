"""
Logging Utilities Module for pulmo-cristal package.

This module provides utilities for setting up and configuring loggers
for the package. It includes functions for creating console loggers,
file loggers, and combined loggers with appropriate formatting.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union, List
from datetime import datetime


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a basic logger with console output.
    
    Args:
        name: Name of the logger
        level: Logging level
        log_format: Format string for log messages
        date_format: Format string for dates in log messages
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Set default formats if not provided
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create console handler with the specified level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(console_handler)
    
    return logger


def add_file_handler(
    logger: logging.Logger,
    log_file: Union[str, Path],
    level: int = logging.INFO,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    mode: str = 'a',
    encoding: str = 'utf-8',
) -> logging.Logger:
    """
    Add a file handler to an existing logger.
    
    Args:
        logger: Existing logger
        log_file: Path to the log file
        level: Logging level for the file handler
        log_format: Format string for log messages
        date_format: Format string for dates in log messages
        mode: File open mode ('a' for append, 'w' for write)
        encoding: File encoding
        
    Returns:
        Logger with file handler added
    """
    # Set default formats if not provided
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    # Ensure directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(log_path, mode=mode, encoding=encoding)
    file_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(file_handler)
    
    return logger


def create_rotating_logger(
    name: str,
    log_dir: Union[str, Path],
    level: int = logging.INFO,
    backup_count: int = 5,
    max_bytes: int = 10_485_760,  # 10 MB
    console_output: bool = True,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> logging.Logger:
    """
    Create a logger with a rotating file handler.
    
    Args:
        name: Name of the logger
        log_dir: Directory for log files
        level: Logging level
        backup_count: Number of backup files to keep
        max_bytes: Maximum size in bytes before rotating files
        console_output: Whether to also log to console
        log_format: Format string for log messages
        date_format: Format string for dates in log messages
        
    Returns:
        Configured logger with rotating file handler
    """
    from logging.handlers import RotatingFileHandler
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Set default formats if not provided
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # Ensure log directory exists
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file path
    log_file = log_dir / f"{name}.log"
    
    # Create rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def create_timed_rotating_logger(
    name: str,
    log_dir: Union[str, Path],
    level: int = logging.INFO,
    when: str = 'midnight',
    interval: int = 1,
    backup_count: int = 14,
    console_output: bool = True,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> logging.Logger:
    """
    Create a logger with a time-based rotating file handler.
    
    Args:
        name: Name of the logger
        log_dir: Directory for log files
        level: Logging level
        when: Rotation interval type ('S', 'M', 'H', 'D', 'W0'-'W6', 'midnight')
        interval: Interval count
        backup_count: Number of backup files to keep
        console_output: Whether to also log to console
        log_format: Format string for log messages
        date_format: Format string for dates in log messages
        
    Returns:
        Configured logger with time-based rotating file handler
    """
    from logging.handlers import TimedRotatingFileHandler
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Set default formats if not provided
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # Ensure log directory exists
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file path
    log_file = log_dir / f"{name}.log"
    
    # Create timed rotating file handler
    file_handler = TimedRotatingFileHandler(
        log_file,
        when=when,
        interval=interval,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def log_to_string(
    level: int = logging.INFO,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> tuple:
    """
    Create a logger that logs to a string buffer.
    
    Args:
        level: Logging level
        log_format: Format string for log messages
        date_format: Format string for dates in log messages
        
    Returns:
        Tuple of (logger, string_io) where string_io can be read for log content
    """
    import io
    
    # Create string buffer
    string_io = io.StringIO()
    
    # Create logger
    logger = logging.getLogger(f"string_logger_{datetime.now().timestamp()}")
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Set default formats if not provided
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # Create string handler
    string_handler = logging.StreamHandler(string_io)
    string_handler.setLevel(level)
    string_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(string_handler)
    
    return logger, string_io


def get_all_loggers() -> List[logging.Logger]:
    """
    Get all loggers that have been created.
    
    Returns:
        List of all existing loggers
    """
    return [logging.getLogger(name) for name in logging.root.manager.loggerDict]


def set_log_level_for_all(level: int) -> None:
    """
    Set the log level for all existing loggers.
    
    Args:
        level: Logging level to set
    """
    for logger in get_all_loggers():
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


def create_debug_logger(name: str) -> logging.Logger:
    """
    Create a logger optimized for debugging with detailed output.
    
    Args:
        name: Name of the logger
        
    Returns:
        Configured debug logger
    """
    return setup_logger(
        name=name,
        level=logging.DEBUG,
        log_format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        date_format="%Y-%m-%d %H:%M:%S.%f"
    )


def create_audit_logger(
    name: str,
    log_dir: Union[str, Path],
    console_output: bool = False,
) -> logging.Logger:
    """
    Create an audit logger for tracking important operations.
    
    Args:
        name: Name of the logger
        log_dir: Directory for audit log files
        console_output: Whether to also log audit events to console
        
    Returns:
        Configured audit logger
    """
    # Create base logger
    logger = logging.getLogger(f"{name}_audit")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create audit-specific format
    log_format = "%(asctime)s - %(levelname)s - [AUDIT] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # Ensure log directory exists
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file path with date in filename
    date_str = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"{name}_audit_{date_str}.log"
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger