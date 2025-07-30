if __name__ == "__main__":
    import logging
    import sys

    from multi_llm_debate.utils.logging_config import setup_logging

    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_all_llm_bar
    from .mad_main import main as mad_main
    from .run_debate import process_llm_bar_dataset
    from .utils import load_llm_bar_dataset

    logger = setup_logging(__name__, log_level=logging.INFO)

    args = Parser(description="Run LLMBar evaluation").parse_args()

    # Check if MAD mode is requested
    use_mad = args.mad or args.task_name == "llm_bar_mad"

    if args.task_name is None:
        task_name = "llm_bar_mad" if use_mad else "llm_bar"
    else:
        task_name = args.task_name

    print(f"Running evaluation for task: {task_name}")
    print(f"Using MAD framework: {use_mad}")

    # Load the dataset
    dataframe = load_llm_bar_dataset(sample_size=args.sample_size)

    if use_mad:
        # Set default config for LLMBar MAD if none provided
        config = args.config
        if config is None and args.config_json is None:
            from pathlib import Path

            config = Path("multi_llm_debate/run/llm_bar/config_gemini.json")
            logger.info(f"Using default config for LLMBar MAD: {config}")

        # Use MAD framework
        mad_main(
            dataframe=dataframe,
            task_name=task_name,
            config=config,
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
        # Use standard debate framework
        shared_main(
            dataframe=dataframe,
            run_debate_fn=process_llm_bar_dataset,
            evaluate_fn=evaluate_all_llm_bar,
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
