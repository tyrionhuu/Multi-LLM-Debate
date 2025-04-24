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


python -m multi_llm_debate.run.big_bench.main \
    --config-json "$CONFIG" \
    --task-name "big_bench" \
    
    # --sample-size 5 \
# Run the evaluation using module path with direct JSON config
python -m multi_llm_debate.run.big_bench.main \
    --config-json "$CONFIG" \
    --task-name "big_bench_pruning" \
    --diversity-pruning "embedding" \
    --diversity-pruning-amount 7 \
    --quality-pruning \
    --quality-pruning-amount 5 \
    