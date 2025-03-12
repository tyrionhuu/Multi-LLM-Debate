#!/bin/bash

# Run the evaluation using module path
python -m multi_llm_debate.run.bool_q.main \
    --config ./multi_llm_debate/configs/11_gemma2_2b.json \
    --sample-size 2000 \
    --max-workers 4
