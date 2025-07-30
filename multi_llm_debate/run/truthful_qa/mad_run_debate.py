import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from multi_llm_debate.utils.logging_config import setup_logging

from ..shared.mad_debate_runner import run_mad_debate_workflow

logger = setup_logging(__name__, log_level=logging.INFO)


def convert_truthful_qa_to_mad_format(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert TruthfulQA data to MAD debate format.

    TruthfulQA format: question, response_A, response_B, response_C, answer (A/B/C)
    MAD format: debate_topic (question with three responses to choose from)

    Args:
        dataframe: TruthfulQA DataFrame with 'question', 'response_A', 'response_B', 'response_C', 'answer' columns

    Returns:
        DataFrame with 'debate_topic' column for MAD
    """
    mad_dataframe = dataframe.copy()

    def create_debate_topic(row):
        question = row["question"]
        response_a = row["response_A"]
        response_b = row["response_B"]
        response_c = row["response_C"]
        correct_answer = row["answer"]

        # For TruthfulQA, we'll create a debate with all three responses
        debate_topic = f"""Question: {question}

Response A: {response_a}

Response B: {response_b}

Response C: {response_c}

Please debate which response (Response A, Response B, or Response C) better answers the question. 
Consider factors such as accuracy, truthfulness, completeness, and helpfulness."""

        return debate_topic

    # Apply the conversion and get debate topics
    debate_topics = mad_dataframe.apply(create_debate_topic, axis=1)

    mad_dataframe["debate_topic"] = debate_topics

    logger.info(f"Converted {len(mad_dataframe)} entries to MAD format")
    return mad_dataframe


def process_truthful_qa_mad_dataset(
    dataframe: pd.DataFrame,
    base_dir: Path = Path("data") / "truthful_qa_mad",
    model_configs: Optional[List[Dict]] = None,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
    quality_pruning_func=None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func=None,
    diversity_pruning_amount: int = 5,
    num_debaters: int = 3,  # Changed from num_players to num_debaters, default to 3 for TruthfulQA (A/B/C)
    provider: str = "google",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_rounds: int = 10,  # Increased default from 3 to 10
    verbose: bool = False,  # Add verbose mode
    task_name: str = "truthful_qa_mad",  # Add task_name parameter
) -> Dict[str, Any]:
    """Process TruthfulQA dataset using MAD framework.

    Args:
        dataframe: TruthfulQA DataFrame
        base_dir: Output directory
        model_configs: Model configurations
        temperature: Temperature for model responses
        max_tokens: Maximum tokens for responses
        batch: Whether to run in batch mode
        batch_size: Batch size
        quality_pruning_func: Quality pruning function
        quality_pruning_amount: Quality pruning amount
        diversity_pruning_func: Diversity pruning function
        diversity_pruning_amount: Diversity pruning amount
        num_debaters: Number of debaters in debate
        provider: LLM provider
        base_url: Base URL for API calls
        api_key: API key for the provider
        max_rounds: Maximum debate rounds

    Returns:
        Execution results dictionary
    """
    logger.info("Converting TruthfulQA format to MAD debate format...")
    mad_dataframe = convert_truthful_qa_to_mad_format(dataframe)

    return run_mad_debate_workflow(
        dataframe=mad_dataframe,
        base_dir=base_dir,
        model_configs=model_configs,
        temperature=temperature,
        max_tokens=max_tokens,
        batch=batch,
        batch_size=batch_size,
        quality_pruning_func=quality_pruning_func,
        quality_pruning_amount=quality_pruning_amount,
        diversity_pruning_func=diversity_pruning_func,
        diversity_pruning_amount=diversity_pruning_amount,
        num_debaters=num_debaters,  # Changed from num_players to num_debaters
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        max_rounds=max_rounds,
        task_name=task_name,  # Pass the actual task name
        verbose=verbose,  # Pass verbose setting
    )


def run_truthful_qa_mad_debate(
    dataframe: pd.DataFrame,
    base_dir: Path = Path("data") / "truthful_qa_mad",
    model_configs: Optional[List[Dict]] = None,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
    quality_pruning_func=None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func=None,
    diversity_pruning_amount: int = 5,
    num_debaters: int = 3,  # Changed from num_players to num_debaters, default to 3 for TruthfulQA (A/B/C)
    provider: str = "google",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_rounds: int = 10,  # Increased default from 3 to 10
    verbose: bool = False,  # Add verbose mode
    task_name: str = "truthful_qa_mad",  # Add task_name parameter
) -> Dict[str, Any]:
    """Run TruthfulQA MAD debate workflow.

    Wrapper function for process_truthful_qa_mad_dataset.

    Args:
        dataframe: TruthfulQA DataFrame
        base_dir: Output directory
        model_configs: Model configurations
        temperature: Temperature for model responses
        max_tokens: Maximum tokens for responses
        batch: Whether to run in batch mode
        batch_size: Batch size
        quality_pruning_func: Quality pruning function
        quality_pruning_amount: Quality pruning amount
        diversity_pruning_func: Diversity pruning function
        diversity_pruning_amount: Diversity pruning amount
        num_debaters: Number of debaters in debate
        provider: LLM provider
        base_url: Base URL for API calls
        api_key: API key for the provider
        max_rounds: Maximum debate rounds

    Returns:
        Execution results dictionary
    """
    return process_truthful_qa_mad_dataset(
        dataframe=dataframe,
        base_dir=base_dir,
        model_configs=model_configs,
        temperature=temperature,
        max_tokens=max_tokens,
        batch=batch,
        batch_size=batch_size,
        quality_pruning_func=quality_pruning_func,
        quality_pruning_amount=quality_pruning_amount,
        diversity_pruning_func=diversity_pruning_func,
        diversity_pruning_amount=diversity_pruning_amount,
        num_debaters=num_debaters,  # Changed from num_players to num_debaters
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        max_rounds=max_rounds,
        verbose=verbose,  # Pass verbose setting
        task_name=task_name,  # Pass task_name parameter
    )
