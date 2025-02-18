import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler to log unhandled exceptions.

    Args:
        exc_type: Type of the exception
        exc_value: Exception instance
        exc_traceback: Traceback object
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def setup_logging(module_name: str, log_level: Optional[int] = None) -> logging.Logger:
    """Set up logging configuration for a module.

    Configures both file and console handlers with formatted output.
    Creates a timestamped log file in the project's logs directory.

    Args:
        module_name: Name of the module requesting logging setup.
            Used as the logger name for hierarchical logging.
        log_level: Optional log level to set for the logger.

    Returns:
        logging.Logger: Configured logger instance with both file and console handlers.
            Will reuse existing logger if one exists for the module name.

    Raises:
        OSError: If unable to create logs directory or log file.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create a unique log file for each run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"debate_{timestamp}.log"

    # Configure logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Get logger
    logger = logging.getLogger(module_name)
    
    # Set default log level to INFO if not specified
    log_level = log_level or logging.INFO
    logger.setLevel(log_level)

    # Add handlers if they haven't been added already
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    # Set up global exception handler
    sys.excepthook = handle_exception

    return logger
