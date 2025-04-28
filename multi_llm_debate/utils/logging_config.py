import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


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


def setup_logging(
    module_name: str,
    log_level: Optional[int] = None,
    propagate: bool = True,
    root_logger_name: str = "multi_llm_debate",
) -> logging.Logger:
    """Set up logging configuration for a module.

    Configures both file and console handlers with formatted output.
    Creates a timestamped log file in the project's logs directory.

    Args:
        module_name: Name of the module requesting logging setup.
            Used as the logger name for hierarchical logging.
        log_level: Optional log level to set for the logger. Defaults to
            logging.WARNING if not specified.
        propagate: Whether to propagate log messages to parent loggers.
            Defaults to True.
        root_logger_name: Name of the root logger for the project. Defaults to
            "multi_llm_debate".

    Returns:
        logging.Logger: Configured logger instance with proper hierarchy setup.

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

    # Set up the root logger
    root_logger = logging.getLogger(root_logger_name)

    # Only add handlers to the root logger if they haven't been added
    if not root_logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Set the root logger level
        root_logger.setLevel(log_level or logging.WARNING)

        # Set up global exception handler
        sys.excepthook = handle_exception

    # Get or create the module's logger, making it a child of the root logger
    if module_name == root_logger_name:
        logger = root_logger
    else:
        qualified_name = (
            f"{root_logger_name}.{module_name}"
            if not module_name.startswith(f"{root_logger_name}.")
            else module_name
        )
        logger = logging.getLogger(qualified_name)

        # Set module-specific log level if provided
        if log_level is not None:
            logger.setLevel(log_level)

        # Control propagation
        logger.propagate = propagate

    return logger
