#!/usr/bin/env python3
"""
Command-line script to run fixed stop analysis on debate results.

Usage:
    python run_fixed_stop_analysis.py [--fixed-round N] [--output-dir DIR]

Examples:
    python run_fixed_stop_analysis.py
    python run_fixed_stop_analysis.py --fixed-round 3
    python run_fixed_stop_analysis.py --fixed-round 5 --output-dir ./results
"""

import argparse
import sys
from pathlib import Path

from .analyze_fixed_stop_table import (
    analyze_fixed_stop_from_table,
    create_detailed_comparison_plot,
    print_detailed_summary,
)


def main():
    """Main function to run fixed stop analysis with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze fixed stop comparison for multi-LLM debate results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--fixed-round",
        type=int,
        default=3,
        help="Round number for fixed stop comparison (default: 3)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="fixed_stop_analysis",
        help="Output directory for results and plots (default: fixed_stop_analysis)"
    )
    
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving results to CSV"
    )
    
    args = parser.parse_args()
    
    print("Fixed Stop Analysis for Multi-LLM Debate")
    print("=" * 50)
    print(f"Fixed round: {args.fixed_round}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    try:
        # Analyze the data
        results_df = analyze_fixed_stop_from_table(fixed_round=args.fixed_round)
        
        # Print summary
        print_detailed_summary(results_df, fixed_round=args.fixed_round)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plots if requested
        if not args.no_plot:
            print(f"\nGenerating plots...")
            create_detailed_comparison_plot(results_df, output_dir, fixed_round=args.fixed_round)
        
        # Save results to CSV if requested
        if not args.no_save:
            csv_path = output_dir / f"fixed_stop_comparison_results_round_{args.fixed_round}.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")
        
        print(f"\nAnalysis complete!")
        print(f"Output directory: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 