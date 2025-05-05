if __name__ == "__main__":
    import logging

    from multi_llm_debate.utils.logging_config import setup_logging

    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_all_mllm_judge_score
    from .run_debate import process_mllm_judge_score_dataset
    from .utils import load_mllm_judge_score_dataset

    logger = setup_logging(__name__, log_level=logging.INFO)

    args = Parser(description="Run MLLM-Judge Score evaluation").parse_args()
    if args.task_name is None:
        task_name = "mllm_judge_score"
    else:
        task_name = args.task_name
    print(f"Running evaluation for task: {task_name}")
    # Load the dataset
    dataframe = load_mllm_judge_score_dataset(sample_size=args.sample_size)
    shared_main(
        dataframe=dataframe,
        run_debate_fn=process_mllm_judge_score_dataset,
        evaluate_fn=evaluate_all_mllm_judge_score,
        task_name=task_name,
        config_json=args.config_json,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch=args.batch,
        quality_pruning_amount=args.quality_pruning_amount,
        quality_pruning_func=args.quality_pruning_func,
        diversity_pruning_func=args.diversity_pruning_func,
        diversity_pruning_amount=args.diversity_pruning_amount,
    )
