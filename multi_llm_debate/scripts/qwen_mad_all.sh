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

$(pwd)/multi_llm_debate/scripts/BIG-Bench/qwen2_5-7b_mad.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/JudgeBench/qwen2_5-7b_mad.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/LLMBar/qwen2_5-7b_mad.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/TruthfulQA/qwen2_5-7b_mad.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/qwen2_5-7b_mad.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/JudgeAnything-pair/qwen2_5-7b_mad.sh -g "$GPU"

