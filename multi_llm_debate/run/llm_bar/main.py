if __name__ == "__main__":
    import logging
    from pathlib import Path

    from multi_llm_debate.utils.logging_config import setup_logging

    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_all_llm_bar
    from .run_debate import process_llm_bar_dataset
    from .utils import load_llm_bar_dataset, preprocess_llm_bar_dataframe

    logger = setup_logging(__name__)
    logger.setLevel(logging.INFO)

    args = Parser(description="Run LLMBar evaluation").parse_args()
    if args.task_name is None:
        task_name = "llm_bar"
    else:
        task_name = args.task_name
    print(f"Running evaluation for task: {task_name}")
    # Load the dataset
    dataset_path = Path("datasets/LLMBar")
    dataframe = load_llm_bar_dataset(
        dataset_path=dataset_path,
    )
    # dataframe = dataframe.sample(10, random_state=42)
    dataframe = preprocess_llm_bar_dataframe(dataframe)

    shared_main(
        dataframe=dataframe,
        run_debate_fn=process_llm_bar_dataset,
        evaluate_fn=evaluate_all_llm_bar,
        task_name=task_name,
        sample_size=args.sample_size,
        config_json=args.config_json,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch=args.batch,
        quality_pruning_amount=args.quality_pruning_amount,
        quality_pruning_func=args.quality_pruning_func,
        diversity_pruning_func=args.diversity_pruning_func,
        diversity_pruning_amount=args.diversity_pruning_amount,
    )
