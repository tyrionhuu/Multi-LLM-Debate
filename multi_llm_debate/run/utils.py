import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple


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
