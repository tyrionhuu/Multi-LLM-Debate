if __name__ == "__main__":
    import logging

    from multi_llm_debate.utils.logging_config import setup_logging

    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_mad_results
    from .run_debate import process_mad_dataset
    from .utils import load_mad_dataset

    logger = setup_logging(__name__, log_level=logging.INFO)

    args = Parser(description="Run MAD (Multi-Agent Debate) evaluation").parse_args()
    logger.info("Parsed arguments: %s", args)

    if args.task_name is None:
        task_name = "mad"
    else:
        task_name = args.task_name

    print(f"Running MAD evaluation for task: {task_name}")

    df = load_mad_dataset(sample_size=args.sample_size)

    shared_main(
        dataframe=df,
        run_debate_fn=process_mad_dataset,
        evaluate_fn=evaluate_mad_results,
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
