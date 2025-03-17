if __name__ == "__main__":
    from pathlib import Path

    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_all_judge_bench
    from .run_debate import run_judge_bench
    from .utils import load_judge_bench_dataset

    args = Parser(description="Run JudgeBench evaluation").parse_args()

    # Load the dataset
    dataset_path = Path("datasets/JudgeBench")
    dataframe = load_judge_bench_dataset(
        dataset_path=dataset_path,
    )
    # dataframe = dataframe.sample(10)
    shared_main(
        dataframe=dataframe,
        run_debate_fn=run_judge_bench,
        evaluate_fn=evaluate_all_judge_bench,
        task_name="JudgeBench",
        sample_size=args.sample_size,
        max_workers=args.max_workers,
        config_path=args.config,
    )
