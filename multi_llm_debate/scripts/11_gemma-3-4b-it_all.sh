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

$(pwd)/multi_llm_debate/scripts/BIG-Bench/11_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/HalluDial/11_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/COMP-Analysis/11_gemma-3-4b-it.sh -g "$GPU"

# $(pwd)/multi_llm_debate/scripts/PRM800K/11_gemma-3-4b-it.sh -g "$GPU"