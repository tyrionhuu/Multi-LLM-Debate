import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from multi_llm_debate.utils.logging_config import setup_logging

from ..shared.mad_debate_runner import run_mad_debate_workflow

logger = setup_logging(__name__, log_level=logging.INFO)


def convert_judge_bench_to_mad_format(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert JudgeBench data to MAD debate format.

    JudgeBench format: question, response_A, response_B, answer (A>B format)
    MAD format: debate_topic (question with two responses to choose from)

    Args:
        dataframe: JudgeBench DataFrame with 'question', 'response_A', 'response_B', 'answer' columns

    Returns:
        DataFrame with 'debate_topic' column for MAD
    """
    mad_dataframe = dataframe.copy()

    def create_debate_topic(row):
        question = row["question"]
        response_a = row["response_A"]
        response_b = row["response_B"]

        # Extract the correct answer from the "A>B" format
        correct_answer = row["answer"]
        if ">" in correct_answer:
            better_response = correct_answer.split(">")[0].strip()
        else:
            # Fallback if format is different
            better_response = correct_answer[0] if correct_answer else "A"

        debate_topic = f"""Question: {question}

Response A: {response_a}

Response B: {response_b}

Please debate which response (Response A or Response B) better answers the question. 
Consider factors such as accuracy, completeness, relevance, helpfulness, and legal reasoning."""

        return debate_topic

    mad_dataframe["debate_topic"] = mad_dataframe.apply(create_debate_topic, axis=1)

    logger.info(f"Converted {len(mad_dataframe)} entries to MAD format")
    return mad_dataframe


def process_judge_bench_mad_dataset(
    dataframe: pd.DataFrame,
    base_dir: Path = Path("data") / "judge_bench_mad",
    model_configs: Optional[List[Dict]] = None,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
    quality_pruning_func=None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func=None,
    diversity_pruning_amount: int = 5,
    num_players: int = 3,
    provider: str = "google",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    """Process JudgeBench dataset using MAD framework.

    Args:
        dataframe: JudgeBench DataFrame
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
        num_players: Number of players in debate
        provider: LLM provider
        base_url: Base URL for API calls
        api_key: API key for the provider
        max_rounds: Maximum debate rounds

    Returns:
        Execution results dictionary
    """
    logger.info("Converting JudgeBench format to MAD debate format...")
    mad_dataframe = convert_judge_bench_to_mad_format(dataframe)

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
        num_players=num_players,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        max_rounds=max_rounds,
    )


def run_judge_bench_mad_debate(
    dataframe: pd.DataFrame,
    base_dir: Path = Path("data") / "judge_bench_mad",
    model_configs: Optional[List[Dict]] = None,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
    quality_pruning_func=None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func=None,
    diversity_pruning_amount: int = 5,
    num_players: int = 3,
    provider: str = "google",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    """Run JudgeBench MAD debate workflow.

    Wrapper function for process_judge_bench_mad_dataset.

    Args:
        dataframe: JudgeBench DataFrame
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
        num_players: Number of players in debate
        provider: LLM provider
        base_url: Base URL for API calls
        api_key: API key for the provider
        max_rounds: Maximum debate rounds

    Returns:
        Execution results dictionary
    """
    return process_judge_bench_mad_dataset(
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
        num_players=num_players,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        max_rounds=max_rounds,
    )
