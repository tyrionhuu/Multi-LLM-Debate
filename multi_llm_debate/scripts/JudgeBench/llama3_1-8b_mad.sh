#!/bin/bash

# Define variables
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_QUANTITY=3
PORT=8000
GPU="0"  # Change this to your GPU ID(s), e.g., "0" for single GPU or "0,1" for 2 GPUs

# Parse command line arguments
SAMPLE_SIZE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --sample-size|-s)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --gpu|-g)
            GPU="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--sample-size|-s SAMPLE_SIZE] [--gpu|-g GPU_ID]"
            echo "Example: $0 --sample-size 100 --gpu 0"
            exit 1
            ;;
    esac
done

echo "Using model: $MODEL_NAME"
echo "GPU: $GPU"
if [[ -n "$SAMPLE_SIZE" ]]; then
    echo "Sample size: $SAMPLE_SIZE"
fi

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up..."
    if [[ -n "$SERVER_PID" ]]; then
        echo "Stopping vLLM server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
    fi
    exit $1
}

# Set up trap to cleanup on exit
trap cleanup EXIT INT TERM

# Check if we're in a Poetry environment
if [[ -z "$POETRY_ACTIVE" ]]; then
    echo "Using Poetry environment..."
    # Poetry environment should already be active if running with poetry run
else
    echo "Poetry environment is already active."
fi

# Start vLLM server
echo "Starting vLLM server..."

if [[ "$GPU" == *","* ]]; then
    # Count the number of GPUs
    IFS=',' read -ra GPU_ARRAY <<< "$GPU"
    if [[ ${#GPU_ARRAY[@]} -eq 2 ]]; then
        echo "Using tensor parallelism with 2 GPUs"
        # Start VLLM server with tensor parallelism
        env CUDA_VISIBLE_DEVICES=$GPU vllm serve $MODEL_NAME --host 0.0.0.0 --port $PORT --max-model-len 64000 --tensor-parallel-size 2 &
    else
        echo "Error: Currently only supporting either 1 GPU or exactly 2 GPUs for tensor parallelism"
        exit 1
    fi
else
    # Single GPU mode
    env CUDA_VISIBLE_DEVICES=$GPU vllm serve $MODEL_NAME --host 0.0.0.0 --port $PORT --max-model-len 32000 --gpu-memory-utilization 0.98 &
fi

SERVER_PID=$!

# Wait for the server to be ready by checking the connection
echo "Waiting for server to start..."
sleep 30
MAX_ATTEMPTS=100
ATTEMPT=2
while ! curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; do
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo "Server did not start after $MAX_ATTEMPTS attempts. Exiting."
        cleanup 1
    fi
    echo "Attempt $ATTEMPT: Server not ready yet. Waiting..."
    sleep 6
    ATTEMPT=$((ATTEMPT+1))
done
echo "Server is ready!"

# Define the configuration as a JSON string for MAD
CONFIG='[[{"name":"'$MODEL_NAME'","provider":"vllm","base_url":"http://localhost:'$PORT'/v1"}]]'

echo "Configuration: $CONFIG"

# Build the command
CMD="python -m multi_llm_debate.run.judge_bench.main --mad --config-json '$CONFIG' --task-name \"judge_bench_mad\""

# Add sample size if specified
if [[ -n "$SAMPLE_SIZE" ]]; then
    CMD="$CMD --sample-size $SAMPLE_SIZE"
fi

# Add batch processing
CMD="$CMD --batch --batch-size 11"

echo "Running MAD JudgeBench evaluation with vLLM..."
echo "Command: $CMD"

# Run the evaluation
eval $CMD

echo "MAD JudgeBench evaluation with vLLM completed!"
cleanup 0 