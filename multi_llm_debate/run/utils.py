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
