#!/bin/bash

# Parse command line arguments
GPU=7  # Default GPU
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu|-g)
            GPU="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gpu|-g GPU_NUMBER]"
            exit 1
            ;;
    esac
done

echo "Using GPU: $GPU"

# Check if Multi-LLM-Debate environment is already activated
if [[ "$CONDA_DEFAULT_ENV" != "Multi-LLM-Debate" ]]; then
    echo "Activating Multi-LLM-Debate conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate Multi-LLM-Debate
else
    echo "Multi-LLM-Debate conda environment is already activated."
fi

# Define variables
MODEL_NAME="/data/share_weight/Qwen2-7B-Instruct"
MODEL_QUANTITY=11
GPU=0
PORT=$((8005 + GPU * 10))

# Start VLLM server with the specified model
export VLLM_LOGGING_LEVEL=ERROR

CUDA_VISIBLE_DEVICES=$GPU vllm serve $MODEL_NAME --host 0.0.0.0 --port $PORT &
SERVER_PID=$!

# Wait for the server to be ready by checking the connection
echo "Waiting for server to start..."
sleep 30
MAX_ATTEMPTS=30
ATTEMPT=1
while ! curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; do
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo "Server did not start after $MAX_ATTEMPTS attempts. Exiting."
        kill $SERVER_PID
        exit 1
    fi
    echo "Attempt $ATTEMPT: Server not ready yet. Waiting..."
    sleep 6
    ATTEMPT=$((ATTEMPT+1))
done
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
python -m multi_llm_debate.run.judge_bench.main \
    --config-json "$CONFIG"

# Kill the VLLM server process when done
kill $SERVER_PID
