#!/bin/bash

# Run from the script directory
cd "$(dirname "$0")"

# Run the evaluation using relative paths
python ../run/bool_q/main.py \
    --config ../configs/3_llama3.json \
    --sample-size 2000 \
    --max-workers 4
