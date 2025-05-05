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

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/5_llama3_1-8b.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/7_llama3_1-8b.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/9_llama3_1-8b.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/7_qwen2_5-7b.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/5_qwen2_5-7b.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/9_qwen2_5-7b.sh -g "$GPU"