if __name__ == "__main__":
    import logging

    from multi_llm_debate.utils.logging_config import setup_logging

    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_all_big_bench
    from .run_debate import process_big_bench_dataset
    from .utils import load_big_bench_dataset

    logger = setup_logging(__name__, log_level=logging.INFO)

    args = Parser(description="Run BIG_Bench evaluation").parse_args()

    if args.task_name is None:
        task_name = "big_bench"
    else:
        task_name = args.task_name

    logger.info(f"Running evaluation for task: {task_name}")
    # Load the dataset
    dataframe = load_big_bench_dataset(sample_size=args.sample_size)

    shared_main(
        dataframe=dataframe,
        run_debate_fn=process_big_bench_dataset,
        evaluate_fn=evaluate_all_big_bench,
        task_name=task_name,
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
