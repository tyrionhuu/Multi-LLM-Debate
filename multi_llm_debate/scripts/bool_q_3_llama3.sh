#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Set paths
CONFIG_PATH="$PROJECT_ROOT/multi_llm_debate/configs/3_llama3.json"
PYTHON_SCRIPT="$PROJECT_ROOT/multi_llm_debate/run/bool_q/main.py"

# Run the evaluation
python "$PYTHON_SCRIPT" \
    --config "$CONFIG_PATH" \
    --sample-size 2000 \
    --max-workers 4
