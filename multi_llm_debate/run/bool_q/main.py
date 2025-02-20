from pathlib import Path
from typing import Optional

from ...utils.download_dataset import load_save_dataset_df
from ..shared.run import main as shared_main
from .evaluate import evaluate_all_bool_q
from .run_debate import run_debate_bool_q
from .utils import process_bool_q_df


def main(
    sample_size: Optional[int] = None,
    max_workers: Optional[int] = 4,
    config_path: Optional[Path] = None,
) -> None:
    """Run boolean question evaluation with configured models."""
    # Load the dataset
    dataset_path = Path("datasets/boolq")
    dataframe = load_save_dataset_df(
        dataset_name="google/boolq",
        dataset_path=dataset_path,
        force_download=False,
    )

    shared_main(
        dataframe=dataframe,
        run_debate_fn=run_debate_bool_q,
        evaluate_fn=evaluate_all_bool_q,
        process_df_fn=process_bool_q_df,
        task_name="bool_q",
        sample_size=sample_size,
        max_workers=max_workers,
        config_path=config_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run boolean question evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config JSON file",
        default=None,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of samples to process",
        default=2000,
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of concurrent workers",
        default=16,
    )

    args = parser.parse_args()
    main(
        sample_size=args.sample_size,
        max_workers=args.max_workers,
        config_path=args.config,
    )
