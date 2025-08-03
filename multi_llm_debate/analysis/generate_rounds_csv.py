#!/usr/bin/env python3
"""
Generate CSV file with correct rates for different benchmarks, models, and rounds.

This script processes multiple benchmarks and models to create a comprehensive CSV file
with columns: benchmark, model_configuration, 1 round, 2 rounds, 3 rounds, 4 rounds

Usage:
    python -m multi_llm_debate.analysis.generate_rounds_csv
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .calculate_correct_rate_by_round import calculate_correct_rate_by_round

# Import utilities for different benchmarks
from ..run.big_bench.utils import (
    compare_big_bench_response,
    extract_0_1_answer,
    load_big_bench_dataset,
)
from ..run.llm_bar.utils import (
    compare_llm_bar_response,
    extract_1_2_answer,
    load_llm_bar_dataset,
)
from ..run.judge_bench.utils import (
    compare_judge_bench_response,
    extract_caption_a_b_answer,
    load_judge_bench_dataset,
)
from ..run.mllm_judge_pair.utils import (
    compare_mllm_judge_pairs_response,
    extract_caption_a_b_answer as extract_mllm_caption_a_b_answer,
    load_mllm_judge_pairs,
)
from ..run.judge_anything_pair.utils import (
    compare_judge_anything_pairs_response,
    extract_caption_a_b_answer as extract_judge_anything_caption_a_b_answer,
    load_judge_anything_pairs_dataset,
)
from ..run.truthful_qa.utils import (
    compare_truthful_qa_response,
    extract_caption_a_b_c_answer,
    load_truthful_qa_dataset,
)

logger = logging.getLogger(__name__)


def get_benchmark_configs() -> Dict[str, Dict]:
    """Get configurations for different benchmarks.
    
    Returns:
        Dictionary mapping benchmark names to their configurations
    """
    return {
        "big_bench": {
            "data_path": "datasets/BIG-Bench/sports_understanding/task.json",
            "data_dir": "data/big_bench",
            "load_func": load_big_bench_dataset,
            "extract_func": extract_0_1_answer,
            "compare_func": compare_big_bench_response,
            "load_kwargs": {"json_path": "datasets/BIG-Bench/sports_understanding/task.json"}
        },
        "llm_bar": {
            "data_path": "datasets/LLMBar",
            "data_dir": "data/llm_bar", 
            "load_func": load_llm_bar_dataset,
            "extract_func": extract_1_2_answer,
            "compare_func": compare_llm_bar_response,
            "load_kwargs": {"dataset_path": "datasets/LLMBar"}
        },
        "judge_bench": {
            "data_path": "datasets/JudgeBench",
            "data_dir": "data/judge_bench",
            "load_func": load_judge_bench_dataset,
            "extract_func": extract_caption_a_b_answer,
            "compare_func": compare_judge_bench_response,
            "load_kwargs": {"dataset_path": "datasets/JudgeBench", "base_path": "."}
        },
        "mllm_judge_pair": {
            "data_path": "datasets/MLLM-Judge/pair_data.tsv",
            "data_dir": "data/mllm_judge_pair",
            "load_func": load_mllm_judge_pairs,
            "extract_func": extract_mllm_caption_a_b_answer,
            "compare_func": compare_mllm_judge_pairs_response,
            "load_kwargs": {"file_path": "datasets/MLLM-Judge/pair_data.tsv"}
        },
        "judge_anything_pair": {
            "data_path": "datasets/JudgeAnything",
            "data_dir": "data/judge_anything_pair",
            "load_func": load_judge_anything_pairs_dataset,
            "extract_func": extract_judge_anything_caption_a_b_answer,
            "compare_func": compare_judge_anything_pairs_response,
            "load_kwargs": {}
        },
        "truthful_qa": {
            "data_path": "datasets/TruthfulQA",
            "data_dir": "data/truthful_qa",
            "load_func": load_truthful_qa_dataset,
            "extract_func": extract_caption_a_b_c_answer,
            "compare_func": compare_truthful_qa_response,
            "load_kwargs": {"dataset_path": "datasets/TruthfulQA"}
        }
    }


def get_model_configurations() -> List[str]:
    """Get list of model configurations to process.
    
    Returns:
        List of model configuration names
    """
    return [
        "Llama-3_1-8B-Instruct(7)",
        "Qwen2_5-7B-Instruct(7)",
        "gemma-3-4b-it(7)",
        "Qwen-2.5-VL-7B(7)",
    ]


def process_benchmark_model(
    benchmark: str,
    model_config: str,
    benchmark_config: Dict,
    max_rounds: int = 4
) -> Optional[Dict]:
    """Process a single benchmark-model combination.
    
    Args:
        benchmark: Name of the benchmark
        model_config: Model configuration name
        benchmark_config: Configuration for the benchmark
        max_rounds: Maximum number of rounds to process
        
    Returns:
        Dictionary with results or None if processing failed
    """
    try:
        # Load dataset
        logger.info(f"Loading {benchmark} dataset...")
        dataframe = benchmark_config["load_func"](**benchmark_config["load_kwargs"])
        
        # Check if model directory exists
        model_dir = Path(benchmark_config["data_dir"]) / model_config
        if not model_dir.exists():
            logger.warning(f"Model directory {model_dir} does not exist, skipping...")
            return None
            
        # Calculate correct rates
        logger.info(f"Processing {benchmark} - {model_config}...")
        results_df = calculate_correct_rate_by_round(
            dataframe=dataframe,
            model_dir=model_dir,
            max_round_number=max_rounds,
            extract_func=benchmark_config["extract_func"],
            compare_func=benchmark_config["compare_func"]
        )
        
        # Extract majority vote results
        majority_results = results_df[results_df["metric"] == "majority"].iloc[0]
        
        # Create result dictionary
        result = {
            "benchmark": benchmark,
            "model_configuration": model_config
        }
        
        # Add round results
        for round_num in range(1, max_rounds + 1):
            round_key = str(round_num)
            if round_key in majority_results:
                result[f"{round_num} rounds"] = majority_results[round_key]
            else:
                result[f"{round_num} rounds"] = "N/A"
                
        return result
        
    except Exception as e:
        logger.error(f"Error processing {benchmark} - {model_config}: {e}")
        return None


def generate_rounds_csv(output_path: str = "rounds_comparison.csv", max_rounds: int = 4) -> None:
    """Generate CSV file with correct rates for all benchmarks and models.
    
    Args:
        output_path: Path to save the CSV file
        max_rounds: Maximum number of rounds to process
    """
    logger.info("Starting rounds comparison CSV generation...")
    
    benchmark_configs = get_benchmark_configs()
    model_configs = get_model_configurations()
    
    all_results = []
    
    # Process each benchmark
    for benchmark, config in benchmark_configs.items():
        logger.info(f"Processing benchmark: {benchmark}")
        
        # Process each model for this benchmark
        for model_config in model_configs:
            result = process_benchmark_model(
                benchmark=benchmark,
                model_config=model_config,
                benchmark_config=config,
                max_rounds=max_rounds
            )
            
            if result is not None:
                all_results.append(result)
    
    # Create DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Reorder columns
        column_order = ["benchmark", "model_configuration"] + [f"{i} rounds" for i in range(1, max_rounds + 1)]
        results_df = results_df[column_order]
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")
        
        # Print summary
        print(f"\nGenerated CSV with {len(results_df)} entries")
        print(f"Benchmarks: {results_df['benchmark'].unique()}")
        print(f"Models: {len(results_df['model_configuration'].unique())}")
        print(f"Rounds: 1-{max_rounds}")
        
        # Show sample of results
        print("\nSample results:")
        print(results_df.head())
        
    else:
        logger.error("No results generated. Check if data files are available.")
        print("No results generated. Please ensure:")
        print("1. Dataset files exist in the correct locations")
        print("2. Model directories exist with debate data")
        print("3. debate_rounds.csv files are available")


def main():
    """Main function to run the rounds CSV generation."""
    logging.basicConfig(level=logging.INFO)
    
    print("Generating Rounds Comparison CSV")
    print("=" * 50)
    
    # Generate CSV with 4 rounds
    generate_rounds_csv("rounds_comparison.csv", max_rounds=4)
    
    print("\nCSV generation complete!")


if __name__ == "__main__":
    main() 