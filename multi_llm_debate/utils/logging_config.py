import logging
from datetime import datetime
from pathlib import Path


def setup_logging(module_name: str) -> logging.Logger:
    """Set up logging configuration for a module.

    Configures both file and console handlers with formatted output.
    Creates a timestamped log file in the project's logs directory.

    Args:
        module_name: Name of the module requesting logging setup.
            Used as the logger name for hierarchical logging.

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
    logger.setLevel(logging.INFO)

    # Add handlers if they haven't been added already
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
