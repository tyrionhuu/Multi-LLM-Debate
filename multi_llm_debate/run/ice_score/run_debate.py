import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ...llm.prompt_builder import PromptBuilder
from ...utils.model_config import ModelConfig
from ..shared.run import process_debate_dataset, process_single_debate_entry
from .prompts import build_ice_score_round_n_prompt, build_ice_score_round_zero_prompt
from .utils import extract_0_4_answer

# Fix the setup_logging call by removing the level parameter
logger = logging.getLogger(__name__)


def process_ice_score_dataset(
    dataframe: pd.DataFrame,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "ice_score",
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    max_workers: Optional[int] = 4,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    parallel: bool = False,
    batch: bool = False,
    quality_pruning_func: Callable = None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func: Callable = None,
    diversity_pruning_amount: int = 5,
) -> Dict[str, Any]:
    """Run the Ice Score task on a DataFrame.

    Args:
        dataframe: Pandas DataFrame containing question, response, id
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files
        model_configs: Optional list of model configurations. If None,
                    default configs will be used.
        overwrite: Whether to overwrite existing debate results (default: False)
        max_workers: Maximum number of concurrent workers (default: 4)
        temperature: Temperature for model responses
        max_tokens: Maximum number of tokens for model responses
        parallel: Whether to run in parallel (default: False)
        batch: Whether to run in batch mode (default: False)
        quality_pruning_func: Optional function for quality pruning
        quality_pruning_amount: Amount for pruning quality
        diversity_pruning_func: Optional function for diversity pruning
        diversity_pruning_amount: Amount for pruning diversity

    Returns:
        Dict containing summary of execution including failed entries

    Raises:
        ValueError: If DataFrame format is invalid
    """
    required_columns = [
        "question",
        "response",
        "id",
    ]

    # Process the dataset for debates
    return process_debate_dataset(
        dataframe=dataframe,
        process_entry_fn=process_ice_score_entry,
        required_columns=required_columns,
        base_dir=base_dir,
        max_rounds=max_rounds,
        model_configs=model_configs,
        overwrite=overwrite,
        max_workers=max_workers,
        task_name="ICE-Score",
        temperature=temperature,
        max_tokens=max_tokens,
        parallel=parallel,
        batch=batch,
        quality_pruning_func=quality_pruning_func,
        quality_pruning_amount=quality_pruning_amount,
        diversity_pruning_func=diversity_pruning_func,
        diversity_pruning_amount=diversity_pruning_amount,
    )


def process_ice_score_entry(
    entry: pd.Series,
    max_rounds: int = 10,
    model_configs: Optional[List[ModelConfig]] = None,
    base_dir: Path = Path("data") / "ice_score",
    overwrite: bool = False,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    parallel: bool = False,
    batch: bool = False,
    quality_pruning_func: Callable = None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func: Callable = None,
    diversity_pruning_amount: int = 5,
) -> Dict[str, Any]:
    """Process a single entry for the ICE-Score task.

    Args:
        entry: Pandas Series containing the entry data
        max_rounds: Current maximum number of rounds
        model_configs: Optional list of model configurations. If None,
                    default configs will be used.
        base_dir: Base directory for output files
        overwrite: Whether to overwrite existing debate results (default: False)
        temperature: Temperature for model responses
        max_tokens: Maximum number of tokens for model responses
        parallel: Whether to run in parallel (default: False)
        batch: Whether to run in batch mode (default: False)
        quality_pruning_func: Optional function for quality pruning
        quality_pruning_amount: Amount for pruning quality
        diversity_pruning_func: Optional function for diversity pruning
        diversity_pruning_amount: Amount for pruning diversity

    Returns:
        Dict containing execution summary including failed entries

    Raises:
        ValueError: If entry format is invalid
    """
    logger.info(f"Processing entry ID: {entry['id']} for ICE-Score task")

    process_single_debate_entry(
        entry=entry,
        required_columns=[
            "question",
            "response",
            "id",
        ],
        base_dir=base_dir,
        max_rounds=max_rounds,
        model_configs=model_configs,
        overwrite=overwrite,
        prompt_builder_fn=lambda prompt_params: PromptBuilder(
            round_zero_fn=build_ice_score_round_zero_prompt,
            round_n_fn=build_ice_score_round_n_prompt,
            prompt_params=prompt_params,
        ),
        prompt_params={
            "question": entry["question"],
            "response": entry["response"],
        },
        extract_func=extract_0_4_answer,
        temperature=temperature,
        max_tokens=max_tokens,
        parallel=parallel,
        batch=batch,
        quality_pruning_func=quality_pruning_func,
        quality_pruning_amount=quality_pruning_amount,
        diversity_pruning_func=diversity_pruning_func,
        diversity_pruning_amount=diversity_pruning_amount,
    )
