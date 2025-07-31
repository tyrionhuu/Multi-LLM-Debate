#!/bin/bash

# KS Threshold Sensitivity Study Runner
# This script provides an easy way to run the KS threshold sensitivity study

set -e  # Exit on any error

# Default values
DATASET_PATH=""
DEBATES_PATH=""
OUTPUT_DIR="output/ks_threshold_sensitivity"
KS_THRESHOLDS="0.01 0.02 0.03 0.05 0.08 0.10 0.15 0.20"
STABILITY_ROUNDS=2
FITTING_METHOD="direct"
MAX_ROUNDS=""
N_RESTARTS=2
VERBOSE=false

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset PATH       Path to the dataset CSV file (required)"
    echo "  -b, --debates PATH       Path to the debates CSV file (required)"
    echo "  -o, --output PATH        Output directory (default: $OUTPUT_DIR)"
    echo "  -t, --thresholds LIST    KS threshold values to test (default: $KS_THRESHOLDS)"
    echo "  -s, --stability N        Number of consecutive stable rounds (default: $STABILITY_ROUNDS)"
    echo "  -f, --fitting METHOD     Fitting method: direct or em (default: $FITTING_METHOD)"
    echo "  -m, --max-rounds N       Maximum rounds to process (default: all)"
    echo "  -r, --restarts N         Number of restarts for fitting (default: $N_RESTARTS)"
    echo "  -v, --verbose            Enable verbose output"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -d data/dataset.csv -b data/debates.csv"
    echo "  $0 -d data/dataset.csv -b data/debates.csv -t '0.01 0.05 0.10' -v"
    echo "  $0 -d data/dataset.csv -b data/debates.csv -o results/ -m 10"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -b|--debates)
            DEBATES_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -t|--thresholds)
            KS_THRESHOLDS="$2"
            shift 2
            ;;
        -s|--stability)
            STABILITY_ROUNDS="$2"
            shift 2
            ;;
        -f|--fitting)
            FITTING_METHOD="$2"
            shift 2
            ;;
        -m|--max-rounds)
            MAX_ROUNDS="$2"
            shift 2
            ;;
        -r|--restarts)
            N_RESTARTS="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$DATASET_PATH" ]]; then
    echo "Error: Dataset path is required"
    print_usage
    exit 1
fi

if [[ -z "$DEBATES_PATH" ]]; then
    echo "Error: Debates path is required"
    print_usage
    exit 1
fi

# Check if files exist
if [[ ! -f "$DATASET_PATH" ]]; then
    echo "Error: Dataset file does not exist: $DATASET_PATH"
    exit 1
fi

if [[ ! -f "$DEBATES_PATH" ]]; then
    echo "Error: Debates file does not exist: $DEBATES_PATH"
    exit 1
fi

# Build the command
CMD="python -m multi_llm_debate.distribution_model.ks_threshold_sensitivity_study"
CMD="$CMD --dataset-path '$DATASET_PATH'"
CMD="$CMD --debates-path '$DEBATES_PATH'"
CMD="$CMD --output-dir '$OUTPUT_DIR'"
CMD="$CMD --ks-thresholds $KS_THRESHOLDS"
CMD="$CMD --stability-rounds $STABILITY_ROUNDS"
CMD="$CMD --fitting-method $FITTING_METHOD"
CMD="$CMD --n-restarts $N_RESTARTS"

if [[ -n "$MAX_ROUNDS" ]]; then
    CMD="$CMD --max-rounds $MAX_ROUNDS"
fi

if [[ "$VERBOSE" == true ]]; then
    CMD="$CMD --verbose"
fi

# Print the command being executed
echo "Running KS threshold sensitivity study..."
echo "Command: $CMD"
echo ""

# Execute the command
eval $CMD

echo ""
echo "Sensitivity study completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - ks_threshold_sensitivity_summary.csv: Summary table"
echo "  - ks_threshold_sensitivity_detailed.json: Detailed results"
echo "  - ks_threshold_sensitivity_summary.png: Summary plots"
echo "  - ks_statistics_evolution.png: KS statistics evolution"
echo "  - rounds_heatmap.png: Rounds processed heatmap"
echo "  - ks_threshold_sensitivity.log: Log file" 