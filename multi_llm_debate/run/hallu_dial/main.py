if __name__ == "__main__":
    import logging
    from pathlib import Path

    from multi_llm_debate.utils.logging_config import setup_logging

    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_all_hallu_dial
    from .run_debate import process_hallu_dial_dataset
    from .utils import load_hallu_dial_dataset

    logger = setup_logging(__name__)
    logger.setLevel(logging.INFO)

    args = Parser(description="Run HalluDial evaluation").parse_args()
    if args.task_name is None:
        task_name = "hallu_dial"
    else:
        task_name = args.task_name
    print(f"Running evaluation for task: {task_name}")
    # Load the dataset
    json_data = Path("datasets/HalluDial/all_train.json")
    dataframe = load_hallu_dial_dataset(
        json_data,
    )

    shared_main(
        dataframe=dataframe,
        run_debate_fn=process_hallu_dial_dataset,
        evaluate_fn=evaluate_all_hallu_dial,
        task_name=task_name,
        sample_size=args.sample_size,
        config_json=args.config_json,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        parallel=args.parallel,
        batch=args.batch,
        quality_pruning_amount=args.quality_pruning_amount,
        quality_pruning_func=args.quality_pruning_func,
        diversity_pruning_func=args.diversity_pruning_func,
        diversity_pruning_amount=args.diversity_pruning_amount,
    )
