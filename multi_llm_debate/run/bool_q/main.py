if __name__ == "__main__":
    from pathlib import Path

    from ...utils.download_dataset import load_save_dataset_df
    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_all_bool_q
    from .run_debate import process_boolean_questions_dataset
    from .utils import process_bool_q_df

    args = Parser(description="Run Bool Q evaluation").parse_args()

    # Load the dataset
    dataset_path = Path("datasets/boolq")
    dataframe = load_save_dataset_df(
        dataset_name="google/boolq",
        dataset_path=dataset_path,
        force_download=False,
    )
    # dataframe = dataframe.sample(10)
    shared_main(
        dataframe=dataframe,
        run_debate_fn=process_boolean_questions_dataset,
        evaluate_fn=evaluate_all_bool_q,
        process_df_fn=process_bool_q_df,
        task_name="bool_q",
        sample_size=args.sample_size,
        config=args.config,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        parallel=args.parallel,
    )
