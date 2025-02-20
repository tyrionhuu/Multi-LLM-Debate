import argparse
import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...utils.logging_config import setup_logging
from ...utils.model_config import ModelConfig

logger = setup_logging(__name__)


@dataclass
class Args:
    """Command line arguments."""

    config: Optional[Path]
    sample_size: int
    max_workers: int


class Parser:
    """Command line argument parser for boolean question evaluation."""

    def __init__(self, description: str = "Run evaluation") -> None:
        """Initialize the parser with boolean question specific arguments.

        Args:
            description: Description for the argument parser
        """
        self.parser = argparse.ArgumentParser(description=description)
        self.parser.add_argument(
            "--config",
            type=Path,
            help="Path to config JSON file",
            default=None,
        )
        self.parser.add_argument(
            "--sample-size",
            type=int,
            help="Number of samples to process",
            default=2000,
        )
        self.parser.add_argument(
            "--max-workers",
            type=int,
            help="Maximum number of concurrent workers",
            default=16,
        )

    def parse_args(self) -> Args:
        """Parse and return the command line arguments.

        Returns:
            Args: Parsed command line arguments.
        """
        return Args(**vars(self.parser.parse_args()))


def format_config_overview(model_configs_list: List[List[ModelConfig]]) -> str:
    """Format model configurations for display in progress bar.

    Args:
        model_configs_list: List of model configuration lists

    Returns:
        str: Formatted string showing number of configs and total models
    """
    total_configs = len(model_configs_list)
    total_models = sum(
        sum(config["quantity"] for config in configs) for configs in model_configs_list
    )
    return f"Running {total_configs} configs ({total_models} total models)"


def build_config_desc(
    model_configs: Optional[List[ModelConfig]], use_cot: bool, max_rounds: int
) -> str:
    """Build a description string for the current model configuration.

    Args:
        model_configs: List of ModelConfig objects
        use_cot: Whether chain-of-thought is enabled
        max_rounds: Maximum number of debate rounds

    Returns:
        A formatted string describing the current configuration
    """
    model_info = []
    total_models = 0

    if model_configs:
        for config in model_configs:
            try:
                name = config["name"]
                quantity = config["quantity"]
                model_info.append(f"{name}×{quantity}")
                total_models += quantity
            except (KeyError, TypeError) as e:
                logger.warning(f"Invalid model config format: {e}")
                continue

    if not model_info:
        model_info = ["default"]
        total_models = 1

    return (
        f"{total_models} models ({', '.join(model_info)}) | "
        f"{'CoT' if use_cot else 'No CoT'} | "
        f"Max rounds: {max_rounds}"
    )


def format_time(seconds: float) -> Tuple[str, str]:
    """Format time in seconds to human readable format and CSV format.

    Args:
        seconds (float): Time in seconds.

    Returns:
        tuple[str, str]: (human readable format, CSV format)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60

    if hours > 0:
        display_time = f"{hours}h {minutes}m {remaining_seconds:.2f}s"
        csv_time = f"{hours}:{minutes:02d}:{remaining_seconds:.2f}"
    elif minutes > 0:
        display_time = f"{minutes}m {remaining_seconds:.2f}s"
        csv_time = f"{minutes}:{remaining_seconds:.2f}"
    else:
        display_time = f"{remaining_seconds:.2f}s"
        csv_time = f"{remaining_seconds:.2f}"

    return display_time, csv_time


def model_configs_to_string(model_configs: List[Dict]) -> str:
    """Convert model configs to a string representation.

    Args:
        model_configs: List of model configuration dictionaries

    Returns:
        str: Formatted string representation sorted by model name and quantity

    Example:
        >>> configs = [
        ...     {"name": "llama2", "quantity": 3},
        ...     {"name": "llama3", "quantity": 3}
        ... ]
        >>> model_configs_to_string(configs)
        'llama2(3)+llama3(3)'
    """
    # Sort configs by model name and quantity
    sorted_configs = sorted(model_configs, key=lambda x: (x["name"], x["quantity"]))

    # Join with plus signs, remove spaces for filesystem safety
    return "+".join(
        f"{config['name']}({config['quantity']})" for config in sorted_configs
    )


def get_latest_round_file(responses_dir: Path) -> Path:
    """Get the file path for the latest debate round.

    Args:
        responses_dir: Directory containing debate round files

    Returns:
        Path to the latest debate round file
    """
    pattern = str(responses_dir / "debate_round_*.json")
    files = glob.glob(pattern)
    if not files:
        raise ValueError(f"No debate round files found in {responses_dir}")

    # Extract round numbers and find max
    rounds = [int(re.search(r"debate_round_(\d+)", f).group(1)) for f in files]
    latest_round = max(rounds)
    return Path(responses_dir / f"debate_round_{latest_round}.json")
