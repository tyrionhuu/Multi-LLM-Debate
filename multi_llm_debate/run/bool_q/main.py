from pathlib import Path

import pandas as pd

from .evaluate import evaluate_df
from .run import run_bool_q
from .utils import process_bool_q_df


def main(test: bool = False) -> None:
    # Load the dataset
    dataset_path = Path("datasets/boolq")
    output_path = Path("data/bool_q/phi3")
    dataframe = pd.read_csv(dataset_path)

    # Process the DataFrame
    processed_dataframe = process_bool_q_df(dataframe)
    if test:
        processed_dataframe = processed_dataframe.head(10)
        
    # Run the Boolean Question task
    run_bool_q(
        dataframe=processed_dataframe,
        base_dir=output_path,
    )

    # Evaluate the results
    accuracy = evaluate_df(output_path, processed_dataframe)
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()