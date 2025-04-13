import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ...llm.prompt_builder import PromptBuilder
from ...utils.logging_config import setup_logging
from ...utils.model_config import ModelConfig
from ..shared.run import process_debate_dataset, process_single_debate_entry
from .prompts import (
    build_truthful_qa_round_n_prompt,
    build_truthful_qa_round_zero_prompt,
)
from .utils import extract_caption_a_b_c_answer

# Fix the setup_logging call by removing the level parameter
logger = setup_logging(__name__)
logger.setLevel(logging.INFO)


def process_truthful_qa_dataset(
    dataframe: pd.DataFrame,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "truthful_qa",
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    max_workers: Optional[int] = 4,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    parallel: bool = False,
    diversity_pruning_func: Callable = None,
    diversity_pruning_amount: int = 5,
) -> Dict[str, Any]:
    """Run the TruthfulQA task on a DataFrame.

    Args:
        dataframe: Pandas DataFrame containing question, answer, passage and id
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files
        use_cot: Whether to use chain-of-thought prompting (default: True)
        model_configs: Optional list of model configurations. If None,
                    default configs will be used.
        overwrite: Whether to overwrite existing debate results (default: False)
        max_workers: Maximum number of concurrent workers (default: 4)
        temperature: Temperature for model responses
        max_tokens: Maximum number of tokens for model responses
        parallel: Whether to run in parallel (default: False)

    Returns:
        Dict containing summary of execution including failed entries

    Raises:
        ValueError: If DataFrame format is invalid
    """
    required_columns = [
        "question",
        "response_A",
        "response_B",
        "response_C",
        "id",
        "answer",
    ]

    return process_debate_dataset(
        dataframe=dataframe,
        process_entry_fn=process_truthful_qa_entry,
        required_columns=required_columns,
        base_dir=base_dir,
        max_rounds=max_rounds,
        model_configs=model_configs,
        overwrite=overwrite,
        max_workers=max_workers,
        task_name="TruthfulQA task",
        temperature=temperature,
        max_tokens=max_tokens,
        parallel=parallel,
        diversity_pruning_func=diversity_pruning_func,
        diversity_pruning_amount=diversity_pruning_amount,
    )


def process_truthful_qa_entry(
    entry: pd.Series,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "truthful_qa",
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    parallel: bool = False,
    diversity_pruning_func: Callable = None,
    diversity_pruning_amount: int = 5,
) -> None:
    """Process a single TruthfulQA entry.

    Args:
        entry: Pandas Series containing the entry data
        max_rounds: Current maximum number of rounds
        base_dir: Base directory for output files
        model_configs: Optional list of model configurations
        overwrite: Whether to overwrite existing debate results
        temperature: Temperature for model responses
        max_tokens: Maximum number of tokens for model responses
        parallel: Whether to run in parallel
        diversity_pruning_func: Optional function for diversity pruning
        diversity_pruning_amount: Amount for pruning diversity
    """

    logger.info(f"Processing entry with ID: {entry['id']}")

    process_single_debate_entry(
        entry=entry,
        required_columns=[
            "question",
            "response_A",
            "response_B",
            "id",
            "response_C",
            "answer",
        ],
        base_dir=base_dir,
        max_rounds=max_rounds,
        model_configs=model_configs,
        overwrite=overwrite,
        prompt_builder_fn=lambda prompt_params: PromptBuilder(
            round_zero_fn=build_truthful_qa_round_zero_prompt,
            round_n_fn=build_truthful_qa_round_n_prompt,
            prompt_params=prompt_params,
        ),
        prompt_params={
            "question": entry["question"],
            "response_a": entry["response_A"],
            "response_b": entry["response_B"],
            "response_c": entry["response_C"],
        },
        extract_func=extract_caption_a_b_c_answer,
        temperature=temperature,
        max_tokens=max_tokens,
        parallel=parallel,
        diversity_pruning_func=diversity_pruning_func,
        diversity_pruning_amount=diversity_pruning_amount,
    )
