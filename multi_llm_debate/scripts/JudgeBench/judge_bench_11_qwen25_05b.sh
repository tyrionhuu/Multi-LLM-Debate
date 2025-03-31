#!/bin/bash

# Run the evaluation using module path
python -m multi_llm_debate.run.judge_bench.main \
    --config ./multi_llm_debate/configs/11_qwen25_05b.json \
    --sample-size 2000 \
    --max-workers 1
