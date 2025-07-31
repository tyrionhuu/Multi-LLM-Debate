#!/usr/bin/env python3
"""
Example script demonstrating KS threshold sensitivity study

This script shows how to use the KS threshold sensitivity study with the JudgeBench dataset.
It provides a complete example of running the sensitivity analysis and interpreting results.

Usage:
    python example_ks_sensitivity.py
"""

import logging
from pathlib import Path

from multi_llm_debate.run.judge_bench.utils import (
    compare_judge_bench_response,
    extract_caption_a_b_answer,
    load_judge_bench_dataset,
)
from multi_llm_debate.utils.model_dir_csv import create_csv_for_all_model_dirs

from .ks_threshold_sensitivity_study import (
    create_summary_dataframe,
    plot_sensitivity_results,
    run_sensitivity_study,
    save_results,
)


def main():
    """Run example KS threshold sensitivity study with JudgeBench dataset."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Define paths
    dataset_path = Path("datasets/JudgeBench")
    debates_path = Path("data/judge_bench/Llama-3_1-8B-Instruct(7)/debate_rounds.csv")
    output_dir = Path("output/ks_threshold_sensitivity_judge_bench_example")

    # Check if required files exist
    if not dataset_path.exists():
        logger.error(
            f"Dataset path {dataset_path} does not exist. Please ensure the JudgeBench dataset is available."
        )
        return

    if not debates_path.exists():
        logger.warning(
            f"Debates path {debates_path} does not exist. Attempting to generate it from JSON files..."
        )
        
        # Get the model directory path
        model_dir = debates_path.parent
        root_data_dir = model_dir.parent  # data/judge_bench
        
        if not model_dir.exists():
            logger.error(
                f"Model directory {model_dir} does not exist. Please ensure the debate data is available."
            )
            return
            
        # Check if there are any JSON files in the model directory
        json_files = list(model_dir.glob("*/debate_round_*.json"))
        if not json_files:
            logger.error(
                f"No debate JSON files found in {model_dir}. Please run the debate process first to generate the data."
            )
            return
            
        logger.info(f"Found {len(json_files)} JSON files. Generating CSV...")
        
        try:
            # Generate CSV for the specific model directory
            create_csv_for_all_model_dirs(root_data_dir)
            
            if not debates_path.exists():
                logger.error(
                    f"Failed to generate {debates_path}. Please check the JSON files and try again."
                )
                return
                
            logger.info(f"Successfully generated {debates_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate CSV file: {e}")
            return

    # Load dataset
    logger.info("Loading JudgeBench dataset...")
    try:
        dataframe = load_judge_bench_dataset(
            dataset_path="datasets/JudgeBench",
            base_path="."
        )
        logger.info(f"Loaded dataset with {len(dataframe)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Define KS thresholds to test
    ks_thresholds = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

    logger.info(f"Running sensitivity study with thresholds: {ks_thresholds}")

    # Run sensitivity study
    results = run_sensitivity_study(
        dataframe=dataframe,
        debates_csv_path=debates_path,
        ks_thresholds=ks_thresholds,
        stability_rounds=2,
        fitting_method="direct",
        max_rounds=10,  # Limit to 10 rounds for faster execution
        n_restarts=2,
        extract_func=extract_caption_a_b_answer,
        compare_func=compare_judge_bench_response,
        verbose=False,  # Set to True for detailed output
    )

    # Save results
    logger.info(f"Saving results to {output_dir}")
    save_results(results, output_dir)

    # Create plots
    logger.info("Creating visualization plots...")
    plot_sensitivity_results(results, output_dir)

    # Print summary
    summary_df = create_summary_dataframe(results)
    print("\n" + "=" * 80)
    print("KS THRESHOLD SENSITIVITY STUDY SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Find optimal threshold
    valid_results = summary_df[summary_df["error"].isna()]
    if not valid_results.empty:
        # Calculate a composite score
        valid_results["score"] = (
            valid_results["convergence_rate"] * 0.7
            + (
                1
                - valid_results["rounds_processed"]
                / valid_results["rounds_processed"].max()
            )
            * 0.3
        )

        optimal_threshold = valid_results.loc[
            valid_results["score"].idxmax(), "ks_threshold"
        ]
        optimal_rounds = valid_results.loc[
            valid_results["score"].idxmax(), "rounds_processed"
        ]
        optimal_convergence = valid_results.loc[
            valid_results["score"].idxmax(), "convergence_rate"
        ]

        print(f"Recommended KS threshold: {optimal_threshold:.3f}")
        print(f"  - Rounds processed: {optimal_rounds}")
        print(f"  - Convergence rate: {optimal_convergence:.3f}")

        # Find most conservative threshold (highest convergence rate)
        conservative_threshold = valid_results.loc[
            valid_results["convergence_rate"].idxmax(), "ks_threshold"
        ]
        conservative_rounds = valid_results.loc[
            valid_results["convergence_rate"].idxmax(), "rounds_processed"
        ]
        conservative_convergence = valid_results.loc[
            valid_results["convergence_rate"].idxmax(), "convergence_rate"
        ]

        print(f"\nMost conservative threshold: {conservative_threshold:.3f}")
        print(f"  - Rounds processed: {conservative_rounds}")
        print(f"  - Convergence rate: {conservative_convergence:.3f}")

        # Find fastest threshold (minimum rounds)
        fastest_threshold = valid_results.loc[
            valid_results["rounds_processed"].idxmin(), "ks_threshold"
        ]
        fastest_rounds = valid_results.loc[
            valid_results["rounds_processed"].idxmin(), "rounds_processed"
        ]
        fastest_convergence = valid_results.loc[
            valid_results["rounds_processed"].idxmin(), "convergence_rate"
        ]

        print(f"\nFastest threshold: {fastest_threshold:.3f}")
        print(f"  - Rounds processed: {fastest_rounds}")
        print(f"  - Convergence rate: {fastest_convergence:.3f}")

        # Show trade-off analysis
        print(f"\n" + "=" * 80)
        print("TRADE-OFF ANALYSIS")
        print("=" * 80)
        print("Lower thresholds (more conservative):")
        print("  + Higher convergence rate")
        print("  + More stable results")
        print("  - More rounds processed")
        print("  - Slower execution")
        print("\nHigher thresholds (more aggressive):")
        print("  + Fewer rounds processed")
        print("  + Faster execution")
        print("  - Lower convergence rate")
        print("  - May miss subtle changes")

    else:
        print("No valid results found. Check the error messages above.")

    logger.info("Example sensitivity study completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("Check the generated plots and CSV files for detailed analysis.")


if __name__ == "__main__":
    main()
