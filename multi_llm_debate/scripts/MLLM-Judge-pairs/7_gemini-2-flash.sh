# Define variables
MODEL_NAME="google/gemini-2.0-flash-001"
MODEL_QUANTITY=7


# Define the configuration as a JSON string
CONFIG='[
    [
        {
            "name": "'$MODEL_NAME'",
            "quantity": '$MODEL_QUANTITY'
        }
    ]
]'


# Run the evaluation using module path with direct JSON config
python -m multi_llm_debate.run.mllm_judge_pair.main \
    --config-json "$CONFIG" \
    --task-name "mllm_judge_pair" \
    --batch \
    --batch-size 11 \
    --sample-size 5
# Run the evaluation using module path with direct JSON config
# python -m multi_llm_debate.run.llm_bar.main \
#     --config-json "$CONFIG" \
#     --task-name "llm_bar_pruning" \
#     --diversity-pruning "answer" \
#     --diversity-pruning-amount 5 \
#     --batch \
#     --batch-size 11 \
#     --sample-size 800