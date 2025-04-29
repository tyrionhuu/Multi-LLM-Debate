if __name__ == "__main__":
    import logging
    from pathlib import Path

    from multi_llm_debate.utils.logging_config import setup_logging

    from ..shared.main import main as shared_main
    from ..shared.utils import Parser
    from .evaluate import evaluate_all_prm800k
    from .run_debate import process_prm800k_dataset
    from .utils import load_prm800k_dataset
    
    logger = setup_logging(__name__, log_level=logging.INFO)
    
    args = Parser(description="Run PRM800K evaluation").parse_args()
    if args.task_name is None:
        task_name = "prm800k"
    else:
        task_name = args.task_name
    print(f"Running evaluation for task: {task_name}")
    dataframe = load_prm800k_dataset()