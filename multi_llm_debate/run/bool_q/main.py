from pathlib import Path

from ...utils.download_dataset import load_save_dataset_df
from ..shared.main import main as shared_main
from ..shared.utils import Parser
from .evaluate import evaluate_all_bool_q
from .run_debate import run_debate_bool_q
from .utils import process_bool_q_df

if __name__ == "__main__":
    args = Parser().parse_args()

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
        sample_size=args.sample_size,
        max_workers=args.max_workers,
        config_path=args.config,
    )
