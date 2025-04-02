#!/bin/bash

# Define the configuration as a JSON string
CONFIG='[
    [
        {
            "name": "/data/share_weight/Qwen2-7B-Instruct",
            "quantity": 11,
            "base_url": "http://localhost:8000/v1"
        }
    ]
]'

# Run the evaluation using module path with direct JSON config
python -m multi_llm_debate.run.judge_bench.main \
    --config "$CONFIG" \
    --sample-size 2000 \
