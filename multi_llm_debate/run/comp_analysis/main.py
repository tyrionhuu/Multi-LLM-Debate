if __name__ == "__main__":
    import logging
    from pathlib import Path

    from multi_llm_debate.utils.logging_config import setup_logging

    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_all_comp_analysis
    from .run_debate import process_comp_analysis_dataset
    from .utils import load_comp_analysis_dataset

    logger = setup_logging(__name__, log_level=logging.INFO)

    args = Parser(description="Run COMP-Analysis evaluation").parse_args()

    if args.task_name is None:
        task_name = "comp_analysis"
    else:
        task_name = args.task_name

    logger.info(f"Running evaluation for task: {task_name}")
    # Load the dataset
    dataframe = load_comp_analysis_dataset()

    shared_main(
        dataframe=dataframe,
        run_debate_fn=process_comp_analysis_dataset,
        evaluate_fn=evaluate_all_comp_analysis,
        task_name=task_name,
        sample_size=args.sample_size,
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
