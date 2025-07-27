import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_mad_dataset(sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load the MAD dataset for evaluation.

    This function creates a sample dataset for MAD evaluation. In a real implementation,
    you would load your actual debate topics from a file or database.

    Args:
        sample_size (Optional[int]): Number of samples to load. If None, loads all.

    Returns:
        pd.DataFrame: DataFrame containing debate topics and metadata.
    """
    # Sample debate topics for evaluation
    sample_topics = [
        "Is artificial intelligence beneficial for society?",
        "Should social media platforms be regulated?",
        "Is remote work more productive than office work?",
        "Should college education be free?",
        "Is climate change the most pressing global issue?",
        "Should autonomous vehicles be allowed on public roads?",
        "Is cryptocurrency a good investment?",
        "Should genetic engineering be allowed in humans?",
        "Is nuclear energy a viable solution for climate change?",
        "Should universal basic income be implemented?",
    ]

    # Create DataFrame
    data = []
    for i, topic in enumerate(sample_topics):
        data.append(
            {
                "id": i,
                "question": topic,
                "category": "debate",
                "difficulty": "medium",
            }
        )

    df = pd.DataFrame(data)

    # Apply sample size if specified
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    logger.info(f"Loaded MAD dataset with {len(df)} samples")
    return df


def create_mad_config(
    debate_topic: str,
    model_configs: list,
    max_rounds: int = 3,
) -> dict:
    """Create a MAD configuration for a given debate topic.

    Args:
        debate_topic (str): The topic to debate
        model_configs (list): List of model configurations
        max_rounds (int): Maximum number of debate rounds

    Returns:
        dict: MAD configuration dictionary
    """
    from multi_llm_debate.mad.prompts import (
        AFFIRMATIVE_PROMPT,
        DEBATE_PROMPT,
        JUDGE_PROMPT_1,
        JUDGE_PROMPT_2,
        MODERATOR_META_PROMPT,
        MODERATOR_PROMPT,
        NEGATIVE_PROMPT,
        PLAYER_META_PROMPT,
    )

    config = {
        "debate_topic": debate_topic,
        "player_meta_prompt": PLAYER_META_PROMPT,
        "moderator_meta_prompt": MODERATOR_META_PROMPT,
        "affirmative_prompt": AFFIRMATIVE_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "moderator_prompt": MODERATOR_PROMPT,
        "judge_prompt_last1": JUDGE_PROMPT_1,
        "judge_prompt_last2": JUDGE_PROMPT_2,
        "debate_prompt": DEBATE_PROMPT,
        "max_rounds": max_rounds,
        "model_configs": model_configs,
    }

    return config


def save_mad_results(results: dict, output_path: Path) -> None:
    """Save MAD debate results to a file.

    Args:
        results (dict): Results from the MAD debate
        output_path (Path): Path to save the results
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved MAD results to {output_path}")


def load_mad_results(input_path: Path) -> dict:
    """Load MAD debate results from a file.

    Args:
        input_path (Path): Path to load the results from

    Returns:
        dict: Loaded results
    """
    with open(input_path, "r") as f:
        results = json.load(f)

    logger.info(f"Loaded MAD results from {input_path}")
    return results
