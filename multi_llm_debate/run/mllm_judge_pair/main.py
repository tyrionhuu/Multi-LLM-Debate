if __name__ == "__main__":
    import logging

    from multi_llm_debate.utils.logging_config import setup_logging

    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_all_mllm_judge_pairs
    from .run_debate import process_mllm_judge_pairs_dataset
    from .utils import load_mllm_judge_pairs
    from .mad_main import main as mad_main

    logger = setup_logging(__name__, log_level=logging.INFO)

    args = Parser(description="Run MLLM-Judge Pairs evaluation").parse_args()

    # Check if MAD mode is requested
    use_mad = args.mad or args.task_name == "mllm_judge_pair_mad"

    if args.task_name is None:
        task_name = "mllm_judge_pair_mad" if use_mad else "mllm_judge_pairs"
    else:
        task_name = args.task_name

    logger.info(f"Running evaluation for task: {task_name}")

    # Load the dataset
    dataframe = load_mllm_judge_pairs(sample_size=args.sample_size)

    if use_mad:
        # Run MAD framework
        mad_main(
            dataframe=dataframe,
            task_name=task_name,
            config=args.config,
            config_json=args.config_json,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            batch=args.batch,
            batch_size=args.batch_size,
            quality_pruning_amount=args.quality_pruning_amount,
            quality_pruning_func=args.quality_pruning_func,
            diversity_pruning_func=args.diversity_pruning_func,
            diversity_pruning_amount=args.diversity_pruning_amount,
        )
    else:
        # Run standard framework
        shared_main(
            dataframe=dataframe,
            run_debate_fn=process_mllm_judge_pairs_dataset,
            evaluate_fn=evaluate_all_mllm_judge_pairs,
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
