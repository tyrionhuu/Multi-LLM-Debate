if __name__ == "__main__":
    from pathlib import Path

    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_all_truthful_qa
    from .run_debate import process_truthful_qa_dataset
    from .utils import load_truthful_qa_dataset, preprocess_dataframe

    args = Parser(description="Run Truthful QA evaluation").parse_args()

    # Load the dataset
    dataset_path = Path("datasets/TruthfulQA")
    dataframe = load_truthful_qa_dataset(
        dataset_path=dataset_path,
    )
    dataframe = preprocess_dataframe(dataframe)
    dataframe = dataframe.sample(100, random_state=42)
    shared_main(
        dataframe=dataframe,
        run_debate_fn=process_truthful_qa_dataset,
        evaluate_fn=evaluate_all_truthful_qa,
        task_name="truthful_qa",
        sample_size=args.sample_size,
        config_json=args.config_json,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        parallel=args.parallel,
    )
