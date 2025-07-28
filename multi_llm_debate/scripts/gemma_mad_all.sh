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

$(pwd)/multi_llm_debate/scripts/TruthfulQA/gemma-3-4b-it_mad.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-pairs/gemma-3-4b-it_mad.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/JudgeAnything-pair/gemma-3-4b-it_mad.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/BIG-Bench/gemma-3-4b-it_mad.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/JudgeBench/gemma-3-4b-it_mad.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/LLMBar/gemma-3-4b-it_mad.sh -g "$GPU"



