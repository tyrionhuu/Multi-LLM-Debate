#!/bin/bash

# Check if Multi-LLM-Debate environment is already activated
if [[ "$CONDA_DEFAULT_ENV" != "Multi-LLM-Debate" ]]; then
    echo "Activating Multi-LLM-Debate conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate Multi-LLM-Debate
else
    echo "Multi-LLM-Debate conda environment is already activated."
fi

# Define variables
MODEL_NAME1="/data/share_weight/Llama-3.1-8B-Instruct"
MODEL_NAME2="/data/share_weight/Qwen2.5-7B-Instruct"
MODEL_QUANTITY1=6
MODEL_QUANTITY2=5
GPU1=1
GPU2=2
PORT1=$((8102 + GPU1 * 10))
PORT2=$((8202 + GPU2 * 10))

export VLLM_LOGGING_LEVEL=ERROR

# Start VLLM server with the specified model
# Setting VLLM_CONFIGURE_LOGGING=0 and adding --max-log-level ERROR to reduce logging
CUDA_VISIBLE_DEVICES=$GPU1 vllm serve $MODEL_NAME1 --host 0.0.0.0 --port $PORT1 --max-model-len 64000 &
SERVER_PID1=$!

CUDA_VISIBLE_DEVICES=$GPU2 vllm serve $MODEL_NAME2 --host 0.0.0.0 --port $PORT2 &
SERVER_PID2=$!

# Wait for the server to be ready by checking the connection
echo "Waiting for server to start..."
sleep 30
MAX_ATTEMPTS=30
ATTEMPT=1
while ! curl -s "http://localhost:${PORT1}/v1/models" > /dev/null 2>&1; do
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo "Server did not start after $MAX_ATTEMPTS attempts. Exiting."
        kill $SERVER_PID1
        exit 1
    fi
    echo "Attempt $ATTEMPT: Server not ready yet. Waiting..."
    sleep 6
    ATTEMPT=$((ATTEMPT+1))
done
echo "Server1 is ready!"

# Wait for the second server to be ready
ATTEMPT=1
while ! curl -s "http://localhost:${PORT2}/v1/models" > /dev/null 2>&1; do
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo "Server did not start after $MAX_ATTEMPTS attempts. Exiting."
        kill $SERVER_PID2
        exit 1
    fi
    echo "Attempt $ATTEMPT: Server not ready yet. Waiting..."
    sleep 6
    ATTEMPT=$((ATTEMPT+1))
done
echo "Server2 is ready!"

# Define the configuration as a JSON string
CONFIG='[
    [
        {
            "name": "'$MODEL_NAME1'",
            "quantity": '$MODEL_QUANTITY1',
            "base_url": "http://localhost:'$PORT1'/v1"
        },
        {
            "name": "'$MODEL_NAME2'",
            "quantity": '$MODEL_QUANTITY2',
            "base_url": "http://localhost:'$PORT2'/v1"
        }
    ]
]'

# Run the evaluation using module path with direct JSON config
python -m multi_llm_debate.run.judge_bench.main \
    --config-json "$CONFIG"

# Kill the VLLM server processes when done
kill $SERVER_PID1
kill $SERVER_PID2
echo "Servers have been killed."
echo "Script completed successfully."
