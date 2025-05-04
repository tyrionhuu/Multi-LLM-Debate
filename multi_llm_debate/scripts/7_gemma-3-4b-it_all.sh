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

$(pwd)/multi_llm_debate/scripts/BIG-Bench/7_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/HalluDial/7_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/COMP-Analysis/7_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/LLMBar/7_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/JudgeBench/7_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/TruthfulQA/7_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/ICE-Score/7_gemma-3-4b-it.sh -g "$GPU"

# $(pwd)/multi_llm_debate/scripts/PRM800K/7_gemma-3-4b-it.sh -g "$GPU"