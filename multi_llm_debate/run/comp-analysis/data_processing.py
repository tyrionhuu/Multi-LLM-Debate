from typing import List
import pandas as pd
FILE_NAMES = [
    "data/comp_analysis/conture-turn_text.txt",
    "data/comp_analysis/fed-turn_text.txt",
    "data/comp_analysis/dailydialog-zhao_text.txt",
    "data/comp_analysis/persona-zhao_text.txt",
    "data/comp_analysis/topical-usr_text.txt",
    "data/comp_analysis/persona-usr_text.txt",
]

def process_files(file_names: List[str] = FILE_NAMES) -> pd.DataFrame:
    """
    Process the given files and return a DataFrame with the content.
    
    Args:
        file_names (List[str]): List of file paths to process.
        
    Returns:
        pd.DataFrame: DataFrame containing id, context, and response columns.
    """
    # Initialize lists to store the context and response data
    contexts = []
    responses = []

    # Iterate over each file and append rows to the text_input list
    for file_name in file_names:
        try:
            with open(file_name, "r") as file:
                # Read each line from the file, split by tabs, and extract context and response
                for line in file:
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            # Get the last two tab-separated values as context and response
                            contexts.append(parts[-2])
                            responses.append(parts[-1])
                        else:
                            print(f"Skipping line with insufficient fields: {line.strip()}")
        except FileNotFoundError:
            print(f"File not found: {file_name}")
        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")
            
    # Create a DataFrame with id, context, and response columns
    df = pd.DataFrame({
        "id": [f"conv_{i}" for i in range(len(contexts))],
        "context": contexts,
        "response": responses
    })
    print(df.head())
    return df

process_files()  # Call the function to process files and print the DataFrame head
