#!/bin/bash

# Define variables
MODEL_NAME="/data/share_weight/Qwen2-7B-Instruct"
MODEL_QUANTITY=11
PORT=8001

# Start VLLM server with the specified model
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_NAME --host 0.0.0.0 --port $PORT &
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
    sleep 5
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
    --config "$CONFIG"

# Kill the VLLM server process when done
kill $SERVER_PID
