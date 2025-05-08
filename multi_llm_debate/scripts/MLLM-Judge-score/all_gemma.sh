
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
            echo "Example: $0 --gpu 0,1 (for tensor parallelism across 2 GPUs)"
            exit 1
            ;;
    esac
done

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-score/3_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-score/5_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-score/7_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-score/9_gemma-3-4b-it.sh -g "$GPU"

$(pwd)/multi_llm_debate/scripts/MLLM-Judge-score/11_gemma-3-4b-it.sh -g "$GPU"