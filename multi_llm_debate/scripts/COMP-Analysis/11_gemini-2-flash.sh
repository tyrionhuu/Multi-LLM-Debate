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


python -m multi_llm_debate.run.comp_analysis.main \
    --config-json "$CONFIG" \
    --task-name "comp_analysis" \
    --batch \
    --batch-size 11 \
    --sample-size 1000 \

# Run the evaluation using module path with direct JSON config
python -m multi_llm_debate.run.comp_analysis.main \
    --config-json "$CONFIG" \
    --task-name "comp_analysis_pruning" \
    --diversity-pruning "answer" \
    --diversity-pruning-amount 7 \
    --batch \
    --batch-size 11 \
    --sample-size 1000 \

echo "BIG-Bench Gemini-2 Flash evaluation completed. "