#!/bin/bash

# Parse -g argument for GPU number
GPU=""
while getopts "g:" opt; do
    case $opt in
        g)
        GPU="$OPTARG"
        ;;
        *)
        ;;
    esac
done

$(pwd)/multi_llm_debate/scripts/BIG-Bench/llama3_1-8b_mad.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/JudgeBench/llama3_1-8b_mad.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/LLMBar/llama3_1-8b_mad.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/TruthfulQA/llama3_1-8b_mad.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/llama3_1-8b_mad.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/JudgeAnything-pair/llama3_1-8b_mad.sh -g "$GPU"

