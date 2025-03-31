#!/bin/bash

# judge_bench_11_gemma2_2b.sh
python -m multi_llm_debate.run.judge_bench.main \
    --config ./multi_llm_debate/configs/11_gemma2_2b.json \
    --sample-size 2000 \
    --max-workers 1

# judge_bench_11_llama2.sh
python -m multi_llm_debate.run.judge_bench.main \
    --config ./multi_llm_debate/configs/11_llama2.json \
    --sample-size 2000 \
    --max-workers 1

# judge_bench_11_llama3.sh
python -m multi_llm_debate.run.judge_bench.main \
    --config ./multi_llm_debate/configs/11_llama3.json \
    --sample-size 2000 \
    --max-workers 4

# judge_bench_11_mistral.sh
python -m multi_llm_debate.run.judge_bench.main \
    --config ./multi_llm_debate/configs/11_mistral.json \
    --sample-size 2000 \
    --max-workers 2

# judge_bench_11_qwen25_05b.sh
python -m multi_llm_debate.run.judge_bench.main \
    --config ./multi_llm_debate/configs/11_qwen25_05b.json \
    --sample-size 2000 \
    --max-workers 2