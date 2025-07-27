#!/bin/bash

# MAD (Multi-Agent Debate) LLMBar Evaluation Script
# This script runs MAD framework evaluation on LLMBar dataset

# Default variables
MODEL_NAME="google/gemini-2.0-flash-001"
MODEL_QUANTITY=7
SAMPLE_SIZE=""
TASK_NAME="llm_bar_mad"
BATCH_SIZE=11
TEMPERATURE=0.7
MAX_TOKENS=2000
NUM_PLAYERS=3
MAX_ROUNDS=3
PROVIDER="google"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            MODEL_NAME="$2"
            shift 2
            ;;
        --quantity|-q)
            MODEL_QUANTITY="$2"
            shift 2
            ;;
        --sample-size|-s)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --task-name|-t)
            TASK_NAME="$2"
            shift 2
            ;;
        --batch-size|-b)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --num-players)
            NUM_PLAYERS="$2"
            shift 2
            ;;
        --max-rounds)
            MAX_ROUNDS="$2"
            shift 2
            ;;
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --help|-h)
            echo "MAD LLMBar Evaluation Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model, -m MODEL        Model name (default: google/gemini-2.0-flash-001)"
            echo "  --quantity, -q NUM       Number of model instances (default: 7)"
            echo "  --sample-size, -s NUM    Sample size for dataset (optional)"
            echo "  --task-name, -t NAME     Task name (default: llm_bar_mad)"
            echo "  --batch-size, -b NUM     Batch size (default: 11)"
            echo "  --temperature NUM        Temperature for generation (default: 0.7)"
            echo "  --max-tokens NUM         Maximum tokens (default: 2000)"
            echo "  --num-players NUM        Number of players in debate (default: 3)"
            echo "  --max-rounds NUM         Maximum debate rounds (default: 3)"
            echo "  --provider NAME          LLM provider (default: google)"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --sample-size 100"
            echo "  $0 --model gemini-1.5-flash --quantity 5 --num-players 5"
            echo "  $0 --provider openai --model gpt-3.5-turbo"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "MAD LLMBar Evaluation Script"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Quantity: $MODEL_QUANTITY"
echo "Provider: $PROVIDER"
echo "Task: $TASK_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Temperature: $TEMPERATURE"
echo "Max tokens: $MAX_TOKENS"
echo "Players: $NUM_PLAYERS"
echo "Max rounds: $MAX_ROUNDS"
if [[ -n "$SAMPLE_SIZE" ]]; then
    echo "Sample size: $SAMPLE_SIZE"
fi
echo "=========================================="

# Check if we're in a Poetry environment
if [[ -z "$POETRY_ACTIVE" ]]; then
    echo "Using Poetry environment..."
    # Poetry environment should already be active if running with poetry run
else
    echo "Poetry environment is already active."
fi

# Define the configuration as a JSON string for MAD
CONFIG='[[{"name":"'$MODEL_NAME'","quantity":'$MODEL_QUANTITY',"provider":"'$PROVIDER'"}]]'

echo "Configuration: $CONFIG"

# Build the command
CMD="python -m multi_llm_debate.run.llm_bar.main"
CMD="$CMD --mad"
CMD="$CMD --config-json \"$CONFIG\""
CMD="$CMD --task-name \"$TASK_NAME\""
CMD="$CMD --batch"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --max-tokens $MAX_TOKENS"

# Add sample size if specified
if [[ -n "$SAMPLE_SIZE" ]]; then
    CMD="$CMD --sample-size $SAMPLE_SIZE"
fi

echo "Running MAD LLMBar evaluation..."
echo "Command: $CMD"

# Run the evaluation
eval $CMD

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "MAD LLMBar evaluation completed successfully!"
    echo "Results saved in: data/$TASK_NAME"
    echo "=========================================="
else
    echo "=========================================="
    echo "MAD LLMBar evaluation failed!"
    echo "=========================================="
    exit 1
fi 