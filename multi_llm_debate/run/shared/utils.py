import argparse
import glob
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from multi_llm_debate.interventions.diversity_pruning import (
    diversity_pruning_by_answer,
    diversity_pruning_by_embedding,
)
from multi_llm_debate.interventions.quality_pruning import quality_pruning
from multi_llm_debate.utils.model_config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class Args:
    """Command line arguments."""

    config: Optional[Path]
    sample_size: int
    config_json: Optional[str]
    temperature: float = 1.0
    max_tokens: int = 6400
    batch: bool = False
    batch_size: int = 11
    quality_pruning: bool = False
    quality_pruning_amount: int = 5
    quality_pruning_func: Callable = None
    diversity_pruning: Optional[str] = None
    diversity_pruning_func: Callable = None
    diversity_pruning_amount: int = 5
    task_name: str = "debate"


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
            "--config-json",
            type=str,
            help="JSON string containing model configurations",
            default=None,
        )
        self.parser.add_argument(
            "--temperature",
            type=float,
            help="Temperature for model responses",
            default=1.0,
        )
        self.parser.add_argument(
            "--max-tokens",
            type=int,
            help="Maximum number of tokens for model responses",
            default=6400,
        )
        self.parser.add_argument(
            "--batch",
            action="store_true",
            default=False,
            help="Enable batch processing",
        )
        self.parser.add_argument(
            "--batch-size",
            type=int,
            help="Size of the batch",
            default=11,
        )
        self.parser.add_argument(
            "--diversity-pruning",
            type=str,
            choices=["embedding", "answer"],
            help="Function for diversity pruning ('embedding' or 'answer')",
            default=None,
        )
        self.parser.add_argument(
            "--diversity-pruning-amount",
            type=int,
            help="Amount for pruning diversity",
            default=5,
        )
        self.parser.add_argument(
            "--task-name",
            type=str,
            help="Name of the task",
            default="debate",
        )
        self.parser.add_argument(
            "--quality-pruning",
            action="store_true",
            help="Enable quality pruning",
        )
        self.parser.add_argument(
            "--quality-pruning-amount",
            type=int,
            help="Amount for pruning quality",
            default=5,
        )

    def parse_args(self) -> Args:
        """Parse and return the command line arguments.

        Returns:
            Args: Parsed command line arguments.
        """
        args: Args = self.parser.parse_args()
        args_dict = vars(args)

        # Convert diversity_pruning string to the corresponding function
        if args.diversity_pruning == "embedding":
            args_dict["diversity_pruning_func"] = diversity_pruning_by_embedding
        elif args.diversity_pruning == "answer":
            args_dict["diversity_pruning_func"] = diversity_pruning_by_answer
        else:
            args_dict["diversity_pruning_func"] = None

        if args.quality_pruning:
            args_dict["quality_pruning_func"] = quality_pruning
        else:
            args_dict["quality_pruning_func"] = None

        return Args(**args_dict)


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


def get_latest_round_file(
    responses_dir: Path, max_rounds: Optional[int] = None
) -> Path:
    """Get the file path for the latest debate round, optionally up to max_rounds.

    Args:
        responses_dir: Directory containing debate round files
        max_rounds: Optional maximum round number to consider

    Returns:
        Path to the latest debate round file
    """
    pattern = str(responses_dir / "debate_round_*.json")
    files = glob.glob(pattern)
    if not files:
        raise ValueError(f"No debate round files found in {responses_dir}")

    # Extract round numbers and filter by max_rounds if provided
    rounds = [int(re.search(r"debate_round_(\d+)", f).group(1)) for f in files]
    if max_rounds is not None:
        rounds = [r for r in rounds if r <= max_rounds - 1]
        if not rounds:
            raise ValueError(
                f"No debate round files <= max_rounds={max_rounds} in {responses_dir}"
            )
    latest_round = max(rounds)
    return Path(responses_dir / f"debate_round_{latest_round}.json")


def model_path_to_model_name(model_path: str) -> str:
    """Convert a model path to a model name.

    Args:
        model_path: Path to the model

    Returns:
        str: Model name with dots replaced by underscores
    """
    # Extract the last part of the path
    last_part = str(model_path).split("/")[-1]
    return last_part.replace(".", "_")


def model_configs_to_string(model_configs: List[Dict]) -> str:
    """Convert model configs to a string representation.

    Args:
        model_configs: List of model configuration dictionaries

    Returns:
        str: Formatted string representation sorted by model name and quantity

    Example:
        >>> configs = [
        ...     {"name": "model/llama2", "quantity": 3},
        ...     {"name": "model/llama3", "quantity": 3}
        ... ]
        >>> model_configs_to_string(configs)
        'llama2(3)+llama3(3)'
    """
    sorted_configs = sorted(
        model_configs,
        key=lambda x: (model_path_to_model_name(x["name"]), x["quantity"]),
    )
    formatted_configs = [
        f"{model_path_to_model_name(config['name'])}({config['quantity']})"
        for config in sorted_configs
    ]
    return "+".join(formatted_configs)


def main():
    file_path = Path("path/to/your/file2.5")
    model_name = model_path_to_model_name(file_path)
    print(f"Model name: {model_name}")


if __name__ == "__main__":
    main()
