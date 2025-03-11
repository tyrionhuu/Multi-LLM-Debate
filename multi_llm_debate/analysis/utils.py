from pathlib import Path


def get_final_round(task_dir: Path) -> int:
    """Gets the final available round number for a given task directory.

    Args:
        task_dir: Path to the task directory.

    Returns:
        int: The highest round number found, or -1 if no rounds exist.
    """
    debate_files = list(task_dir.glob("debate_round_*.json"))
    if not debate_files:
        return -1
    
    round_numbers = [
        int(f.stem.split('_')[-1]) for f in debate_files
    ]
    return max(round_numbers)
