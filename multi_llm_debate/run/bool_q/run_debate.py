from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ...llm.prompts import build_bool_q_round_n_prompt, build_bool_q_round_zero_prompt
from ...utils.logging_config import setup_logging
from ...utils.model_config import ModelConfig
from ..shared.run_debate import run_debate, run_debate_single_entry

logger = setup_logging(__name__)


def get_bool_q_prompt_params(entry: pd.Series) -> Dict[str, Any]:
    """Extract prompt parameters from a boolean question entry."""
    return {
        "question": entry["question"],
        "passage": entry["passage"],
    }

def run_debate_bool_q(
    dataframe: pd.DataFrame,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "bool_q",
    use_cot: bool = True,
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    max_workers: Optional[int] = 4,
) -> Dict[str, Any]:
    """Run the Boolean Question task on a DataFrame.

    Args:
        dataframe: Pandas DataFrame containing question, answer, passage and id
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files
        use_cot: Whether to use chain-of-thought prompting (default: True)
        model_configs: Optional list of model configurations. If None,
                    default configs will be used.
        overwrite: Whether to overwrite existing debate results (default: False)
        max_workers: Maximum number of concurrent workers (default: 4)

    Returns:
        Dict containing summary of execution including failed entries

    Raises:
        ValueError: If DataFrame format is invalid
    """
    return run_debate(
        dataframe=dataframe,
        round_zero_fn=build_bool_q_round_zero_prompt,
        round_n_fn=build_bool_q_round_n_prompt,
        required_columns=["question", "answer", "passage", "id"],
        get_prompt_params=get_bool_q_prompt_params,
        max_rounds=max_rounds,
        base_dir=base_dir,
        use_cot=use_cot,
        model_configs=model_configs,
        overwrite=overwrite,
        max_workers=max_workers,
    )


def run_debate_bool_q_single_entry(
    entry: pd.Series,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "bool_q",
    use_cot: bool = True,
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    max_workers: Optional[int] = 4,
) -> None:
    """Run a single entry for the Boolean Question task.

    Args:
        entry: Pandas Series containing question, answer, passage and id
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files
        use_cot: Whether to use chain-of-thought prompting
        model_configs: Optional list of model configurations
        overwrite: Whether to overwrite existing debate results
        max_workers: Maximum number of concurrent workers

    Raises:
        ValueError: If entry format is invalid
        RuntimeError: If debate execution fails
    """
    required_fields = ["question", "answer", "passage", "id"]
    missing_fields = [field for field in required_fields if field not in entry]
    if missing_fields:
        logger.error(f"Missing required fields: {missing_fields}")
        raise ValueError(
            "Entry must contain 'question', 'answer', 'passage', and 'id'."
        )

    prompt_params = {
        "question": entry["question"],
        "passage": entry["passage"],
    }

    run_debate_single_entry(
        entry=entry,
        round_zero_fn=build_bool_q_round_zero_prompt,
        round_n_fn=build_bool_q_round_n_prompt,
        prompt_params=prompt_params,
        max_rounds=max_rounds,
        base_dir=base_dir,
        use_cot=use_cot,
        model_configs=model_configs,
        overwrite=overwrite,
        max_workers=max_workers,
    )
