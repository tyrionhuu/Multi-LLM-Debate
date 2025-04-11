if __name__ == "__main__":
    from pathlib import Path

    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_all_judge_bench
    from .run_debate import process_judge_bench_dataset
    from .utils import load_judge_bench_dataset

    args = Parser(description="Run JudgeBench evaluation").parse_args()

    # Load the dataset
    dataset_path = Path("datasets/JudgeBench")
    dataframe = load_judge_bench_dataset(
        dataset_path=dataset_path,
    )

    shared_main(
        dataframe=dataframe,
        run_debate_fn=process_judge_bench_dataset,
        evaluate_fn=evaluate_all_judge_bench,
        task_name="judge_bench",
        sample_size=args.sample_size,
        config_json=args.config_json,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        parallel=args.parallel,
        diversity_pruning_func=args.diversity_pruning_func,
        pruning_amount=args.pruning_amount,
    )
