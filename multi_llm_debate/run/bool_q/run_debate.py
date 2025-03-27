from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ...debate.agents_ensemble import AgentsEnsemble
from ...debate.debate import debate
from ...llm.prompt_builder import PromptBuilder
from ...utils.logging_config import setup_logging
from ...utils.model_config import ModelConfig
from ..shared.run import run_debate_task
from .prompts import build_bool_q_round_n_prompt, build_bool_q_round_zero_prompt
from .utils import extract_bool_answer
from ..shared.run import run_single_entry

logger = setup_logging(__name__)


def run_bool_q(
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
    required_columns = ["question", "answer", "passage", "id"]

    return run_debate_task(
        dataframe=dataframe,
        process_entry_fn=run_bool_q_single_entry,
        required_columns=required_columns,
        base_dir=base_dir,
        max_rounds=max_rounds,
        use_cot=use_cot,
        model_configs=model_configs,
        overwrite=overwrite,
        max_workers=max_workers,
        task_name="Boolean Question task",
    )


def run_bool_q_single_entry(
    entry: pd.Series,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "bool_q",
    use_cot: bool = True,
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    max_workers: Optional[int] = 4,
) -> None:
    """Run a single entry for the Boolean Question task."""
    run_single_entry(
        entry=entry,
        required_columns=["question", "answer", "passage", "id"],
        base_dir=base_dir,
        max_rounds=max_rounds,
        use_cot=use_cot,
        model_configs=model_configs,
        overwrite=overwrite,
        max_workers=max_workers,
        prompt_builder_fn=lambda prompt_params: PromptBuilder(
            round_zero_fn=build_bool_q_round_zero_prompt,
            round_n_fn=build_bool_q_round_n_prompt,
            prompt_params=prompt_params,
        ),
        prompt_params={
            "question": entry["question"],
            "passage": entry["passage"],
            "use_cot": use_cot,
        },
        process_answer_fn=extract_bool_answer,
    )
