if __name__ == "__main__":
    from pathlib import Path

    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_llm_bar_responses
    from .run_debate import process_llm_bar_dataset
    from .utils import load_llm_bar_dataset, preprocess_llm_bar_dataframe

    args = Parser(description="Run LLMBar evaluation").parse_args()

    # Load the dataset
    dataset_path = Path("datasets/LLMBar")
    dataframe = load_llm_bar_dataset(
        dataset_path=dataset_path,
    )
    dataframe = dataframe.sample(10, random_state=42)
    dataframe = preprocess_llm_bar_dataframe(dataframe)

    shared_main(
        dataframe=dataframe,
        run_debate_fn=process_llm_bar_dataset,
        evaluate_fn=evaluate_llm_bar_responses,
        task_name="llm_bar",
        sample_size=args.sample_size,
        config_json=args.config_json,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        parallel=args.parallel,
    )
