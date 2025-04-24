# Define variables
MODEL_NAME="google/gemini-2.0-flash-001"
MODEL_QUANTITY=11


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
python -m multi_llm_debate.run.ice_score.main \
    --config-json "$CONFIG" \
    --task-name "ice_score" \
    
# Run the evaluation using module path with direct JSON config
python -m multi_llm_debate.run.ice_score.main \
    --config-json "$CONFIG" \
    --task-name "ice_score_pruning" \
    --quality-pruning \
    --diversity-pruning "embedding" \
    --pruning-amount 5 \

    