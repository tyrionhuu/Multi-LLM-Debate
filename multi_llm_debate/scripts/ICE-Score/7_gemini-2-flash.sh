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
CUDA_VISIBLE_DEVICES=all python -m multi_llm_debate.run.ice_score.main \
    --config-json "$CONFIG" \
    --task-name "ice_score" \
    --batch \
    --batch-size 11 \
# Run the evaluation using module path with direct JSON config
# CUDA_VISIBLE_DEVICES=all python -m multi_llm_debate.run.ice_score.main \
#     --config-json "$CONFIG" \
#     --task-name "ice_score_pruning" \
#     --diversity-pruning "answer" \
#     --diversity-pruning-amount 5 \
#     --batch \
#     --batch-size 11 \
    