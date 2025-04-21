#!/bin/bash

# Import utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
source "$PROJECT_ROOT/multi_llm_debate/scripts/utils/shell_utils.sh"

# Parse command line arguments
GPU="7"  # Default GPU
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu|-g)
            GPU="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gpu|-g GPU_NUMBER(S)]"
            echo "Example: $0 --gpu 0,1 (for tensor parallelism across 2 GPUs)"
            exit 1
            ;;
    esac
done

echo "Using GPU(s): $GPU"

# Use imported function to activate conda environment
activate_conda_env "Multi-LLM-Debate"

# Define cleanup wrapper that uses the imported function
cleanup() {
    cleanup_vllm_server "$SERVER_PID"
    exit ${1:-0}
}

# Set trap to catch exit signals
trap cleanup SIGINT SIGTERM EXIT

# Define variables
MODEL_NAME="/data/share_weight/gemma-3-4b-it"
MODEL_QUANTITY=11
# For port, use the first GPU in case of multiple GPUs
FIRST_GPU=$(echo $GPU | cut -d',' -f1)
PORT=$((8002 + FIRST_GPU * 10))

export VLLM_LOGGING_LEVEL=ERROR

# Check if we have multiple GPUs and set tensor parallelism accordingly
if [[ "$GPU" == *","* ]]; then
    # Count the number of GPUs
    IFS=',' read -ra GPU_ARRAY <<< "$GPU"
    if [[ ${#GPU_ARRAY[@]} -eq 2 ]]; then
        echo "Using tensor parallelism with 2 GPUs"
        # Start VLLM server with tensor parallelism
        CUDA_VISIBLE_DEVICES=$GPU vllm serve $MODEL_NAME --host 0.0.0.0 --port $PORT --max-model-len 64000 --tensor-parallel-size 2 &
    else
        echo "Error: Currently only supporting either 1 GPU or exactly 2 GPUs for tensor parallelism"
        exit 1
    fi
else
    # Single GPU mode
    CUDA_VISIBLE_DEVICES=$GPU vllm serve $MODEL_NAME --host 0.0.0.0 --port $PORT --max-model-len 32000 &
fi

SERVER_PID=$!

# Wait for the server to be ready by checking the connection
wait_for_server "$PORT" 30 6

echo "Server is ready!"

# Define the configuration as a JSON string
CONFIG='[
    [
        {
            "name": "'$MODEL_NAME'",
            "quantity": '$MODEL_QUANTITY',
            "base_url": "http://localhost:'$PORT'/v1"
        }
    ]
]'

# Run the evaluation using module path with direct JSON config
python -m multi_llm_debate.run.ice_score.main \
    --sample-size 5 \
    --config-json "$CONFIG"  

# Run the evaluation using module path with direct JSON config
# python -m multi_llm_debate.run.ice_score.main \
#     --config-json "$CONFIG" \
#     --task-name "ice_score_pruning" \
#     --quality-pruning \
#     --diversity-pruning "embedding" \
#     --pruning-amount 5 \

cleanup 1
