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
python -m multi_llm_debate.run.hallu_dial.main \
    --config-json "$CONFIG" \
    --task-name "hallu_dial" \

# Run the evaluation using module path with direct JSON config
python -m multi_llm_debate.run.hallu_dial.main \
    --config-json "$CONFIG" \
    --task-name "hallu_dial_pruning" \
    --diversity-pruning "embedding" \
    --diversity-pruning-amount 7 \
    --quality-pruning \
    --quality-pruning-amount 5 \
    