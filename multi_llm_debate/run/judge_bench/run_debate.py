import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ...llm.prompt_builder import PromptBuilder
from ...utils.logging_config import setup_logging
from ...utils.model_config import ModelConfig
from ..shared.run import process_debate_dataset, process_single_debate_entry
from .prompts import (
    build_judge_bench_round_n_prompt,
    build_judge_bench_round_zero_prompt,
)
from .utils import extract_caption_a_b_answer

# Fix the setup_logging call by removing the level parameter
logger = setup_logging(__name__)
logger.setLevel(logging.INFO)


def process_judge_bench_dataset(
    dataframe: pd.DataFrame,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "judge_bench",
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    max_workers: Optional[int] = 4,
) -> Dict[str, Any]:
    """Run the JudgeBench task on a DataFrame.

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
    required_columns = ["question", "response_A", "response_B", "id"]

    return process_debate_dataset(
        dataframe=dataframe,
        process_entry_fn=process_judge_bench_entry,
        required_columns=required_columns,
        base_dir=base_dir,
        max_rounds=max_rounds,
        model_configs=model_configs,
        overwrite=overwrite,
        max_workers=max_workers,
        task_name="JudgeBench task",
    )


def process_judge_bench_entry(
    entry: pd.Series,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "judge_bench",
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    parallel: bool = False,
) -> None:
    """Process a single JudgeBench entry."""
    process_single_debate_entry(
        entry=entry,
        required_columns=["question", "response_A", "response_B", "id"],
        base_dir=base_dir,
        max_rounds=max_rounds,
        model_configs=model_configs,
        overwrite=overwrite,
        prompt_builder_fn=lambda prompt_params: PromptBuilder(
            round_zero_fn=build_judge_bench_round_zero_prompt,
            round_n_fn=build_judge_bench_round_n_prompt,
            prompt_params=prompt_params,
        ),
        prompt_params={
            "question": entry["question"],
            "response_a": entry["response_A"],
            "response_b": entry["response_B"],
        },
        process_answer_fn=extract_caption_a_b_answer,
        temperature=temperature,
        max_tokens=max_tokens,
        parallel=parallel,
    )
