from .utils import process_bool_q_df
from .run import run_bool_q
from .evaluate import evaluate_df
from pathlib import Path
import pandas as pd
def main():
    # Load the dataset
    dataset_path = Path("data" / "bool_q" / "dataset.csv")
    dataframe = pd.read_csv(dataset_path)

    # Process the DataFrame
    processed_dataframe = process_bool_q_df(dataframe)

    # Run the Boolean Question task
    run_bool_q(processed_dataframe)

    # Evaluate the results
    response_base_dir = Path("data" / "bool_q" / "responses")
    accuracy = evaluate_df(response_base_dir, processed_dataframe)
    print(f"Accuracy: {accuracy:.2f}")