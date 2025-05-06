#!/bin/bash

# Parse -g argument for GPU number
GPU=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu|-g)
            GPU="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gpu|-g GPU_NUMBER(S)]"
            exit 1
            ;;
    esac
done

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/5_llama3_1-8b.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/7_llama3_1-8b.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/9_llama3_1-8b.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-score/5_llama3_1-8b.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-score/7_llama3_1-8b.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-score/9_llama3_1-8b.sh -g "$GPU"