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
MODEL_NAME="/data/share_weight/Llama-3.1-8B-Instruct"
MODEL_QUANTITY=11
PORT=8002

# Start VLLM server with the specified model
# Setting VLLM_CONFIGURE_LOGGING=0 and adding --max-log-level ERROR to reduce logging
vllm serve $MODEL_NAME --host 0.0.0.0 --port $PORT --max-model-len 64000 &
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
