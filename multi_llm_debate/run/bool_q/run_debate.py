from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ...llm.prompt_builder import PromptBuilder
from ...llm.prompts import build_bool_q_round_n_prompt, build_bool_q_round_zero_prompt
from ...utils.logging_config import setup_logging
from ...utils.model_config import ModelConfig
from ..shared.run_debate import run_debate

logger = setup_logging(__name__)


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
    # Initialize prompt builder with bool_q specific prompts
    prompt_builder = PromptBuilder(
        round_zero_fn=build_bool_q_round_zero_prompt,
        round_n_fn=build_bool_q_round_n_prompt,
        prompt_params={},  # Will be updated per entry
    )

    # Required columns for bool_q task
    required_columns = ["question", "answer", "passage", "id"]

    return run_debate(
        dataframe=dataframe,
        prompt_builder=prompt_builder,
        required_columns=required_columns,
        max_rounds=max_rounds,
        base_dir=base_dir,
        use_cot=use_cot,
        model_configs=model_configs,
        overwrite=overwrite,
        max_workers=max_workers,
    )
