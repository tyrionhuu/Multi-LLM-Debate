#!/usr/bin/env python3
"""
Sensitivity Study for KS Threshold in Adaptive Stopping

This script performs a comprehensive sensitivity analysis of the ks_threshold parameter
used in adaptive stopping for distribution fitting. It tests various threshold values
and compares the results to understand how the threshold affects:

1. Number of rounds processed before stopping
2. Final distribution parameters
3. Convergence behavior
4. Computational efficiency

Usage:
    python ks_threshold_sensitivity_study.py --dataset-path <path> --debates-path <path> [options]
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from .fit_distribution_adaptive_stopping import analyze_distributions_adaptive_stopping


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("ks_threshold_sensitivity.log"),
        ],
    )


def run_sensitivity_study(
    dataframe: pd.DataFrame,
    debates_csv_path: Path,
    ks_thresholds: List[float],
    stability_rounds: int = 2,
    fitting_method: str = "direct",
    max_rounds: Optional[int] = None,
    n_restarts: int = 2,
    extract_func=None,
    compare_func=None,
    verbose: bool = False,
) -> Dict:
    """
    Run sensitivity study across different KS threshold values.

    Args:
        dataframe: Input DataFrame with task data
        debates_csv_path: Path to debates CSV file
        ks_thresholds: List of KS threshold values to test
        stability_rounds: Number of consecutive stable rounds required
        fitting_method: Method for fitting distributions
        max_rounds: Maximum rounds to process (None for all)
        n_restarts: Number of restarts for fitting
        extract_func: Function to extract responses
        compare_func: Function to compare responses
        verbose: Whether to print detailed output

    Returns:
        Dictionary containing results for each threshold
    """
    results = {}

    for threshold in tqdm(ks_thresholds, desc="Testing KS thresholds"):
        logging.info(f"Testing ks_threshold = {threshold}")

        try:
            # Run analysis with current threshold
            aggregated_df, fit_results, ks_statistics, chi_test_results = (
                analyze_distributions_adaptive_stopping(
                    dataframe=dataframe,
                    debates_csv_path=debates_csv_path,
                    fitting_method=fitting_method,
                    max_rounds=max_rounds,
                    n_restarts=n_restarts,
                    verbose=verbose,
                    enforce_increasing_success=False,
                    extract_func=extract_func,
                    compare_func=compare_func,
                    adaptive_stopping=True,
                    ks_threshold=threshold,
                    stability_rounds=stability_rounds,
                )
            )

            # Extract key metrics
            rounds_processed = len(fit_results)
            final_ks_stat = ks_statistics[-1] if len(ks_statistics) > 1 else None
            stopped_early = (
                rounds_processed < len(aggregated_df)
                if max_rounds is None
                else rounds_processed < max_rounds
            )

            # Get final distribution parameters
            final_params = fit_results[-1] if fit_results else None

            # Calculate convergence metrics
            convergence_metrics = calculate_convergence_metrics(
                ks_statistics, threshold
            )

            results[threshold] = {
                "rounds_processed": rounds_processed,
                "total_rounds_available": len(aggregated_df),
                "stopped_early": stopped_early,
                "final_ks_statistic": final_ks_stat,
                "final_params": final_params,
                "ks_statistics": ks_statistics,
                "chi_test_results": chi_test_results,
                "convergence_metrics": convergence_metrics,
                "aggregated_df": aggregated_df,
                "fit_results": fit_results,
            }

            logging.info(f"  Rounds processed: {rounds_processed}")
            logging.info(f"  Stopped early: {stopped_early}")
            logging.info(
                f"  Final KS stat: {final_ks_stat:.4f}"
                if final_ks_stat
                else "  Final KS stat: None"
            )

        except Exception as e:
            logging.error(f"Error testing threshold {threshold}: {e}")
            results[threshold] = {
                "error": str(e),
                "rounds_processed": 0,
                "stopped_early": False,
                "final_ks_statistic": None,
                "final_params": None,
                "ks_statistics": [],
                "chi_test_results": [],
                "convergence_metrics": {},
                "aggregated_df": pd.DataFrame(),
                "fit_results": [],
            }

    return results


def calculate_convergence_metrics(
    ks_statistics: List[Optional[float]], threshold: float
) -> Dict:
    """
    Calculate convergence-related metrics from KS statistics.

    Args:
        ks_statistics: List of KS statistics per round
        threshold: KS threshold used

    Returns:
        Dictionary with convergence metrics
    """
    # Remove None values (first round)
    valid_ks_stats = [ks for ks in ks_statistics if ks is not None]

    if not valid_ks_stats:
        return {
            "mean_ks": None,
            "max_ks": None,
            "min_ks": None,
            "std_ks": None,
            "rounds_below_threshold": 0,
            "convergence_rate": 0.0,
        }

    metrics = {
        "mean_ks": np.mean(valid_ks_stats),
        "max_ks": np.max(valid_ks_stats),
        "min_ks": np.min(valid_ks_stats),
        "std_ks": np.std(valid_ks_stats),
        "rounds_below_threshold": sum(1 for ks in valid_ks_stats if ks < threshold),
        "convergence_rate": sum(1 for ks in valid_ks_stats if ks < threshold)
        / len(valid_ks_stats),
    }

    return metrics


def create_summary_dataframe(results: Dict) -> pd.DataFrame:
    """
    Create a summary DataFrame from sensitivity study results.

    Args:
        results: Results dictionary from sensitivity study

    Returns:
        Summary DataFrame
    """
    summary_data = []

    for threshold, result in results.items():
        if "error" in result:
            summary_data.append(
                {
                    "ks_threshold": threshold,
                    "rounds_processed": 0,
                    "stopped_early": False,
                    "final_ks_statistic": None,
                    "mean_ks": None,
                    "max_ks": None,
                    "min_ks": None,
                    "std_ks": None,
                    "rounds_below_threshold": 0,
                    "convergence_rate": 0.0,
                    "error": result["error"],
                }
            )
        else:
            conv_metrics = result["convergence_metrics"]
            summary_data.append(
                {
                    "ks_threshold": threshold,
                    "rounds_processed": result["rounds_processed"],
                    "stopped_early": result["stopped_early"],
                    "final_ks_statistic": result["final_ks_statistic"],
                    "mean_ks": conv_metrics["mean_ks"],
                    "max_ks": conv_metrics["max_ks"],
                    "min_ks": conv_metrics["min_ks"],
                    "std_ks": conv_metrics["std_ks"],
                    "rounds_below_threshold": conv_metrics["rounds_below_threshold"],
                    "convergence_rate": conv_metrics["convergence_rate"],
                    "error": None,
                }
            )

    return pd.DataFrame(summary_data)


def plot_sensitivity_results(results: Dict, output_dir: Path) -> None:
    """
    Create comprehensive plots for sensitivity study results.

    Args:
        results: Results dictionary from sensitivity study
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create summary DataFrame
    summary_df = create_summary_dataframe(results)

    # Set up the plotting style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # 1. Rounds processed vs KS threshold
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Filter out error cases
    valid_df = summary_df[summary_df["error"].isna()]

    if not valid_df.empty:
        # Plot 1: Rounds processed
        axes[0, 0].plot(
            valid_df["ks_threshold"],
            valid_df["rounds_processed"],
            "o-",
            linewidth=2,
            markersize=8,
        )
        axes[0, 0].set_xlabel("KS Threshold")
        axes[0, 0].set_ylabel("Rounds Processed")
        axes[0, 0].set_title("Rounds Processed vs KS Threshold")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Final KS statistic
        axes[0, 1].plot(
            valid_df["ks_threshold"],
            valid_df["final_ks_statistic"],
            "s-",
            linewidth=2,
            markersize=8,
        )
        axes[0, 1].set_xlabel("KS Threshold")
        axes[0, 1].set_ylabel("Final KS Statistic")
        axes[0, 1].set_title("Final KS Statistic vs KS Threshold")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Convergence rate
        axes[1, 0].plot(
            valid_df["ks_threshold"],
            valid_df["convergence_rate"],
            "^-",
            linewidth=2,
            markersize=8,
        )
        axes[1, 0].set_xlabel("KS Threshold")
        axes[1, 0].set_ylabel("Convergence Rate")
        axes[1, 0].set_title("Convergence Rate vs KS Threshold")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Mean KS statistic
        axes[1, 1].plot(
            valid_df["ks_threshold"],
            valid_df["mean_ks"],
            "d-",
            linewidth=2,
            markersize=8,
        )
        axes[1, 1].set_xlabel("KS Threshold")
        axes[1, 1].set_ylabel("Mean KS Statistic")
        axes[1, 1].set_title("Mean KS Statistic vs KS Threshold")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "ks_threshold_sensitivity_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Detailed KS statistics evolution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Select a few representative thresholds for detailed view
    representative_thresholds = sorted(
        [t for t in results.keys() if "error" not in results[t]]
    )[:4]

    for i, threshold in enumerate(representative_thresholds):
        row, col = i // 2, i % 2
        result = results[threshold]
        ks_stats = [ks for ks in result["ks_statistics"] if ks is not None]

        if ks_stats:
            rounds = list(range(2, len(ks_stats) + 2))  # Start from round 2
            axes[row, col].plot(
                rounds,
                ks_stats,
                "o-",
                linewidth=2,
                markersize=6,
                label=f"Threshold: {threshold}",
            )
            axes[row, col].axhline(
                y=threshold,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Threshold ({threshold})",
            )
            axes[row, col].set_xlabel("Round Number")
            axes[row, col].set_ylabel("KS Statistic")
            axes[row, col].set_title(
                f"KS Statistics Evolution (Threshold: {threshold})"
            )
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "ks_statistics_evolution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. Heatmap of rounds processed
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create heatmap data
    threshold_values = sorted([t for t in results.keys() if "error" not in results[t]])
    rounds_data = [results[t]["rounds_processed"] for t in threshold_values]

    # Create a 2D array for heatmap (single row)
    heatmap_data = np.array(rounds_data).reshape(1, -1)

    sns.heatmap(
        heatmap_data,
        xticklabels=[f"{t:.3f}" for t in threshold_values],
        yticklabels=["Rounds"],
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        ax=ax,
    )
    ax.set_xlabel("KS Threshold")
    ax.set_title("Rounds Processed for Different KS Thresholds")

    plt.tight_layout()
    plt.savefig(output_dir / "rounds_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_results(results: Dict, output_dir: Path) -> None:
    """
    Save sensitivity study results to files.

    Args:
        results: Results dictionary from sensitivity study
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary DataFrame
    summary_df = create_summary_dataframe(results)
    summary_df.to_csv(output_dir / "ks_threshold_sensitivity_summary.csv", index=False)

    # Save detailed results (excluding large DataFrames)
    detailed_results = {}
    for threshold, result in results.items():
        detailed_result = result.copy()
        # Remove large objects for JSON serialization
        if "aggregated_df" in detailed_result:
            del detailed_result["aggregated_df"]
        if "fit_results" in detailed_result:
            # Keep only essential parameters
            detailed_result["fit_results"] = [
                {
                    "round_number": fr.get("round_number"),
                    "w": fr.get("w"),
                    "alpha1": fr.get("alpha1"),
                    "beta1": fr.get("beta1"),
                    "alpha2": fr.get("alpha2"),
                    "beta2": fr.get("beta2"),
                    "log_likelihood": fr.get("log_likelihood"),
                    "n_iter": fr.get("n_iter"),
                }
                for fr in result["fit_results"]
            ]
        detailed_results[threshold] = detailed_result

    with open(output_dir / "ks_threshold_sensitivity_detailed.json", "w") as f:
        json.dump(detailed_results, f, indent=2, default=str)


def main():
    """Main function to run the sensitivity study."""
    parser = argparse.ArgumentParser(description="KS Threshold Sensitivity Study")
    parser.add_argument(
        "--dataset-path", type=Path, required=True, help="Path to the dataset CSV file"
    )
    parser.add_argument(
        "--debates-path", type=Path, required=True, help="Path to the debates CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/ks_threshold_sensitivity"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--ks-thresholds",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
        help="List of KS threshold values to test",
    )
    parser.add_argument(
        "--stability-rounds",
        type=int,
        default=2,
        help="Number of consecutive stable rounds required",
    )
    parser.add_argument(
        "--fitting-method",
        type=str,
        default="direct",
        choices=["direct", "em"],
        help="Fitting method to use",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum rounds to process (None for all)",
    )
    parser.add_argument(
        "--n-restarts", type=int, default=2, help="Number of restarts for fitting"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Load dataset
    logging.info(f"Loading dataset from {args.dataset_path}")
    try:
        dataframe = pd.read_csv(args.dataset_path)
        logging.info(f"Loaded dataset with {len(dataframe)} rows")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # Run sensitivity study
    logging.info("Starting KS threshold sensitivity study")
    logging.info(f"Testing thresholds: {args.ks_thresholds}")

    results = run_sensitivity_study(
        dataframe=dataframe,
        debates_csv_path=args.debates_path,
        ks_thresholds=args.ks_thresholds,
        stability_rounds=args.stability_rounds,
        fitting_method=args.fitting_method,
        max_rounds=args.max_rounds,
        n_restarts=args.n_restarts,
        verbose=args.verbose,
    )

    # Save results
    logging.info(f"Saving results to {args.output_dir}")
    save_results(results, args.output_dir)

    # Create plots
    logging.info("Creating plots")
    plot_sensitivity_results(results, args.output_dir)

    # Print summary
    summary_df = create_summary_dataframe(results)
    print("\n" + "=" * 80)
    print("KS THRESHOLD SENSITIVITY STUDY SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Find optimal threshold (minimum rounds with good convergence)
    valid_results = summary_df[summary_df["error"].isna()]
    if not valid_results.empty:
        # Simple heuristic: prefer thresholds that stop early but have good convergence
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
        print(f"\nRecommended KS threshold: {optimal_threshold:.3f}")

    logging.info("Sensitivity study completed successfully!")


if __name__ == "__main__":
    main()
