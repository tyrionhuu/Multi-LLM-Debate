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

$(pwd)/multi_llm_debate/scripts/BIG-Bench/3_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/HalluDial/3_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/COMP-Analysis/3_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/LLMBar/3_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/JudgeBench/3_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/TruthfulQA/3_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/ICE-Score/3_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/3_gemma-3-4b-it.sh -g "$GPU"

# $(pwd)/multi_llm_debate/scripts/PRM800K/3_gemma-3-4b-it.sh -g "$GPU"