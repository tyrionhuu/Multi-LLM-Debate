import json
import re
from pathlib import Path
from typing import List, Literal, Union

import pandas as pd

DATA_PATHS = [
    "datasets/COMP-Analysis/turn_level_texts/conture-turn_text.txt",
    "datasets/COMP-Analysis/turn_level_texts/fed-turn_text.txt",
    "datasets/COMP-Analysis/turn_level_texts/dailydialog-zhao_text.txt",
    "datasets/COMP-Analysis/turn_level_texts/persona-zhao_text.txt",
    "datasets/COMP-Analysis/turn_level_texts/topical-usr_text.txt",
    "datasets/COMP-Analysis/turn_level_texts/persona-usr_text.txt",
]


def load_comp_analysis_dataset(
    data_paths: Union[str, Path, List[Union[str, Path]]] = DATA_PATHS,
    template: str = "{}\t{}",
) -> pd.DataFrame:
    """
    Load multiple text files into a DataFrame, processing each line by extracting
    the last two tab-separated fields and formatting them.

    Args:
        data_paths (Union[str, Path, List[Union[str, Path]]]): Path(s) to the text files.
        template (str): Template string to format the extracted fields.

    Returns:
        pd.DataFrame: DataFrame containing the processed data from the text files,
            with columns 'input' and 'answer'.
    """
    if isinstance(data_paths, (str, Path)):
        data_paths = [data_paths]

    data = []
    for path in data_paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    fields = line.strip().split("\t")
                    if len(fields) < 2:
                        continue  # skip lines that don't have at least two fields
                    formatted = template.format(fields[-2], fields[-1])
                    data.append(formatted)

    df = pd.DataFrame(data, columns=["input_response"])
    # Optionally, split into 'input' and 'response' columns if needed
    df[["input", "response"]] = df["input_response"].str.split("\t", n=1, expand=True)
    df = df.drop(columns=["input_response"])
    df.insert(0, "id", range(len(df)))  # Add id column as the first column
    return df


if __name__ == "__main__":
    data = load_comp_analysis_dataset()
    print(data)
