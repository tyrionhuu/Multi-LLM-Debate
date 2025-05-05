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


python -m multi_llm_debate.run.truthful_qa.main \
    --config-json "$CONFIG" \
    --task-name "truthful_qa" \
    --batch \
    --batch-size 11 \

# # Run the evaluation using module path with direct JSON config
python -m multi_llm_debate.run.truthful_qa.main \
    --config-json "$CONFIG" \
    --task-name "truthful_qa_pruning" \
    --diversity-pruning "answer" \
    --diversity-pruning-amount 5 \
    --batch \
    --batch-size 11 \

# python -m multi_llm_debate.run.truthful_qa.main \
#     --config-json "$CONFIG" \
#     --task-name "truthful_qa_pruning_all" \
#     --diversity-pruning "answer" \
#     --diversity-pruning-amount 5 \
#     --batch \
#     --batch-size 11 \
#     --quality-pruning \
#     --quality-pruning-amount 7 \