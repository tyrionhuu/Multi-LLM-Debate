import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from multi_llm_debate.utils.logging_config import setup_logging

from ..shared.mad_debate_runner import save_mad_results_to_csv
from ..shared.utils import Parser
from .mad_evaluate import evaluate_all_truthful_qa_mad
from .mad_run_debate import run_truthful_qa_mad_debate
from .utils import load_truthful_qa_dataset

logger = setup_logging(__name__, log_level=logging.INFO)


def main(
    dataframe: pd.DataFrame,
    task_name: str = "truthful_qa_mad",
    config: Optional[Union[Path, List[Dict]]] = None,
    config_json: Optional[str] = None,
    run_debate: bool = True,
    run_evaluation: bool = True,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
    quality_pruning_func: Optional[Callable] = None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func: Optional[Callable] = None,
    diversity_pruning_amount: int = 5,
    num_debaters: int = 3,  # Changed from num_players to num_debaters, default to 3 for TruthfulQA (A/B/C)
    provider: str = "google",  # Changed from ollama to google
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_rounds: int = 10,  # Increased default from 3 to 10
    verbose: bool = False,  # Add verbose mode
) -> None:
    """Run MAD debate evaluation on TruthfulQA dataset with configured models.

    Args:
        dataframe: Input DataFrame containing the TruthfulQA data
        task_name: Name of the debate task (default: "truthful_qa_mad")
        config: Path to JSON config file or list of model configurations
        config_json: JSON string containing model configurations
        run_debate: Whether to run the debate or just evaluate existing results
        run_evaluation: Whether to run evaluation after debate execution
        temperature: Temperature for model responses
        max_tokens: Maximum number of tokens for model responses
        batch: Whether to run in batch mode
        batch_size: Size of the batch.
        quality_pruning_func: Function for quality pruning
        quality_pruning_amount: Amount of pruning to apply
        diversity_pruning_func: Function for diversity pruning
        diversity_pruning_amount: Amount of pruning to apply
        num_debaters: Number of debaters in the debate
        provider: LLM provider
        base_url: Base URL for API calls
        api_key: API key for the provider
        max_rounds: Maximum number of debate rounds
    """

    if not run_debate:
        logger.info("Skipping debate execution as run_debate=False")
        return

    # Parse model configurations
    model_configs_list = []

    if config_json is not None:
        import json

        try:
            model_configs_list = json.loads(config_json)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string provided in config_json")
    elif isinstance(config, list):
        model_configs_list = [config]
    else:
        # Use provided config path or default to config_gemini.json in task directory
        if config is None:
            config = Path(f"multi_llm_debate/run/truthful_qa/config_gemini.json")

        # Load configuration from file
        import json

        with open(config) as f:
            loaded_config = json.load(f)
            # Wrap in list if it's not already a list of lists
            if (
                isinstance(loaded_config, list)
                and loaded_config
                and isinstance(loaded_config[0], dict)
            ):
                model_configs_list = [loaded_config]
            else:
                model_configs_list = loaded_config

    # Run MAD debates for each model configuration
    for i, model_configs in enumerate(model_configs_list):
        logger.info(
            f"Running MAD debate with model config {i+1}/{len(model_configs_list)}"
        )

        try:
            results = run_truthful_qa_mad_debate(
                dataframe=dataframe,
                base_dir=Path(f"data/{task_name}"),
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
            )

            # Print execution summary
            print(f"\nMAD Debate Execution Summary for {task_name}:")
            print("-" * 50)
            print(f"Total entries processed: {results['total_entries']}")
            print(f"Successfully processed: {results['processed_count']}")
            print(f"Failed entries: {len(results['failed_entries'])}")
            print(f"Success rate: {results['success_rate']:.2f}%")
            print(f"Number of debaters: {results['num_debaters']}")
            print(f"Provider: {results['provider']}")
            print(f"Max rounds: {results['max_rounds']}")

            # Run evaluation on the results if requested
            if run_evaluation:
                print(f"\nRunning evaluation for {task_name}...")
                # Convert the original dataframe to MAD format to get the _correct_is_1 flag
                from .mad_run_debate import convert_truthful_qa_to_mad_format

                mad_dataframe = convert_truthful_qa_to_mad_format(dataframe)

                evaluation_results = evaluate_all_truthful_qa_mad(
                    base_dir=Path(f"data/{task_name}"),
                    original_dataframe=mad_dataframe,
                    model_configs=model_configs,
                )

                # Save results to CSV
                # Use a reasonable default running time since we don't track it precisely
                save_mad_results_to_csv(
                    evaluation_results=evaluation_results,
                    task_name=task_name,
                    model_configs=model_configs,
                    report_path=Path(f"data/{task_name}"),
                    running_time=0.0,  # We'll calculate this properly later
                )
            else:
                print(f"\nSkipping evaluation as run_evaluation=False")

        except Exception as e:
            logger.error(f"Error running MAD debate with config {i+1}: {str(e)}")
            raise


if __name__ == "__main__":
    args = Parser(description="Run TruthfulQA MAD evaluation").parse_args()

    if args.task_name is None or args.task_name == "default_task":
        task_name = "truthful_qa_mad"
    else:
        task_name = args.task_name

    print(f"Running MAD evaluation for task: {task_name}")

    # Load the dataset
    dataframe = load_truthful_qa_dataset(sample_size=args.sample_size)

    main(
        dataframe=dataframe,
        task_name=task_name,
        config_json=args.config_json,
        run_evaluation=True,  # Always run evaluation by default
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch=args.batch,
        batch_size=args.batch_size,
        quality_pruning_amount=args.quality_pruning_amount,
        quality_pruning_func=args.quality_pruning_func,
        diversity_pruning_func=args.diversity_pruning_func,
        diversity_pruning_amount=args.diversity_pruning_amount,
        # MAD-specific parameters (you can add these to the Parser if needed)
        num_debaters=3,  # Changed from num_players to num_debaters, default to 3 for TruthfulQA (A/B/C)
        provider="google",  # Use Google provider for Gemini models
        max_rounds=10,  # Increased default from 3 to 10
        verbose=args.verbose,  # Pass verbose from command line arguments
    )
