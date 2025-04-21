if __name__ == "__main__":
    from pathlib import Path

    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_all_ice_score
    from .run_debate import process_ice_score_dataset
    from .utils import load_ice_score_dataset

    args = Parser(description="Run ICE-Score evaluation").parse_args()
    if args.task_name is None:
        task_name = "ice_score"
    else:
        task_name = args.task_name
    print(f"Running evaluation for task: {task_name}")
    # Load the dataset
    json_data = Path("datasets/ICE-Score/")
    dataframe = load_ice_score_dataset(
        dataset_path=json_data,
    )

    shared_main(
        dataframe=dataframe,
        run_debate_fn=process_ice_score_dataset,
        evaluate_fn=evaluate_all_ice_score,
        task_name=task_name,
        sample_size=args.sample_size,
        config_json=args.config_json,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        parallel=args.parallel,
        quality_pruning_amount=args.quality_pruning_amount,
        quality_pruning_func=args.quality_pruning_func,
        diversity_pruning_func=args.diversity_pruning_func,
        diversity_pruning_amount=args.diversity_pruning_amount,
    )
