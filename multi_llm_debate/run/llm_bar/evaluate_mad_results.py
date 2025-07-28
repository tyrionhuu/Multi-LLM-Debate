#!/usr/bin/env python3

import logging
from pathlib import Path

from multi_llm_debate.utils.logging_config import setup_logging

from .mad_evaluate import evaluate_all_llm_bar_mad
from .utils import load_llm_bar_dataset


def main():
    """Evaluate MAD results against LLMBar ground truth."""
    logger = setup_logging(__name__, log_level=logging.INFO)

    # Load the original dataset with only the entries we have results for
    dataframe = load_llm_bar_dataset(sample_size=10)  # Only evaluate first 10 entries

    # Define the base directory where MAD results are stored
    base_dir = Path("data/llm_bar_mad")

    # Run evaluation
    evaluation_results = evaluate_all_llm_bar_mad(
        base_dir=base_dir,
        original_dataframe=dataframe,
        model_configs=[{"name": "google/gemini-2.0-flash-001", "provider": "google"}],
    )

    return evaluation_results


if __name__ == "__main__":
    main()
