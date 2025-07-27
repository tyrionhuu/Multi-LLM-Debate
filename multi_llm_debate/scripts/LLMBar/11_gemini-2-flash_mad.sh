#!/bin/bash

# Define variables
MODEL_NAME="google/gemini-2.0-flash-001"
MODEL_QUANTITY=11

# Parse command line arguments
SAMPLE_SIZE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --sample-size|-s)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--sample-size|-s SAMPLE_SIZE]"
            echo "Example: $0 --sample-size 100"
            exit 1
            ;;
    esac
done

echo "Using model: $MODEL_NAME"
echo "Model quantity: $MODEL_QUANTITY"
if [[ -n "$SAMPLE_SIZE" ]]; then
    echo "Sample size: $SAMPLE_SIZE"
fi

# Check if we're in a Poetry environment
if [[ -z "$POETRY_ACTIVE" ]]; then
    echo "Using Poetry environment..."
    # Poetry environment should already be active if running with poetry run
else
    echo "Poetry environment is already active."
fi

# Define the configuration as a JSON string for MAD
CONFIG='[[{"name":"'$MODEL_NAME'","quantity":'$MODEL_QUANTITY',"provider":"google"}]]'

echo "Configuration: $CONFIG"

# Build the command
CMD="python -m multi_llm_debate.run.llm_bar.main --mad --config-json \"$CONFIG\" --task-name \"llm_bar_mad\""

# Add sample size if specified
if [[ -n "$SAMPLE_SIZE" ]]; then
    CMD="$CMD --sample-size $SAMPLE_SIZE"
fi

# Add batch processing
CMD="$CMD --batch --batch-size 11"

echo "Running MAD LLMBar evaluation..."
echo "Command: $CMD"

# Run the evaluation
eval $CMD

echo "MAD LLMBar evaluation completed!" 