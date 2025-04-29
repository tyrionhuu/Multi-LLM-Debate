if __name__ == "__main__":
    import logging
    import os
    from pathlib import Path

    import pandas as pd

    from multi_llm_debate.utils.logging_config import setup_logging

    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_all_truthful_qa
    from .run_debate import process_truthful_qa_dataset
    from .utils import load_truthful_qa_dataset

    logger = setup_logging(__name__, log_level=logging.INFO)

    args = Parser(description="Run Truthful QA evaluation").parse_args()
    logger.info("Parsed arguments: %s", args)
    df_path = Path("output/truthful_qa/processed_data.csv")

    if args.task_name is None:
        task_name = "truthful_qa"
    else:
        task_name = args.task_name

    print(f"Running evaluation for task: {task_name}")

    if not df_path.exists():
        df = load_truthful_qa_dataset(dataset_path="datasets/TruthfulQA")
        os.makedirs("output/truthful_qa", exist_ok=True)
        df.to_csv(df_path, index=False)
    else:
        df = pd.read_csv(df_path)
    
    shared_main(
        dataframe=df,
        run_debate_fn=process_truthful_qa_dataset,
        evaluate_fn=evaluate_all_truthful_qa,
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
