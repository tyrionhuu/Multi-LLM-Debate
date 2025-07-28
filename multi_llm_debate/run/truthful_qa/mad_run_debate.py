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
    MAD format: debate_topic (question with two responses to choose from)

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

        # For TruthfulQA, we'll create a debate between the correct answer and one incorrect answer
        # We'll randomly choose which incorrect answer to use for variety
        import random

        if correct_answer == "A":
            correct_response = response_a
            incorrect_responses = [response_b, response_c]
        elif correct_answer == "B":
            correct_response = response_b
            incorrect_responses = [response_a, response_c]
        else:  # correct_answer == 'C'
            correct_response = response_c
            incorrect_responses = [response_a, response_b]

        # Randomly choose one incorrect response
        incorrect_response = random.choice(incorrect_responses)

        # Randomly assign correct and incorrect to Response 1 or Response 2
        if random.choice([True, False]):
            response_1 = correct_response
            response_2 = incorrect_response
            correct_is_1 = True
        else:
            response_1 = incorrect_response
            response_2 = correct_response
            correct_is_1 = False

        debate_topic = f"""Question: {question}

Response 1: {response_1}

Response 2: {response_2}

Please debate which response (Response 1 or Response 2) better answers the question. 
Consider factors such as accuracy, truthfulness, completeness, and helpfulness."""

        # Store which response is correct for evaluation
        row["_correct_is_1"] = correct_is_1

        return debate_topic

    mad_dataframe["debate_topic"] = mad_dataframe.apply(create_debate_topic, axis=1)

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
    num_players: int = 3,
    provider: str = "google",
    max_rounds: int = 3,
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
        num_players: Number of players in debate
        provider: LLM provider
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
        num_players=num_players,
        provider=provider,
        max_rounds=max_rounds,
        task_name="truthful_qa",
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
    num_players: int = 3,
    provider: str = "google",
    max_rounds: int = 3,
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
        num_players: Number of players in debate
        provider: LLM provider
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
        num_players=num_players,
        provider=provider,
        max_rounds=max_rounds,
    )
