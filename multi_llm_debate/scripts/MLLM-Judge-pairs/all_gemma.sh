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

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/3_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/5_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/7_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/9_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/11_gemma-3-4b-it.sh -g "$GPU"