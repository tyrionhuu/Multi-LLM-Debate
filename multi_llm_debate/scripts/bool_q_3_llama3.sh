#!/bin/bash

# Run from the script directory
cd "$(dirname "$0")"

# Run the evaluation using module path
python -m multi_llm_debate.run.bool_q.main \
    --config multi_llm_debate/configs/3_llama3.json \
    --sample-size 2000 \
    --max-workers 4
