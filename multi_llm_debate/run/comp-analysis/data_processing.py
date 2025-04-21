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
        pd.DataFrame: DataFrame containing the processed content.
    """
    # Initialize an empty list to store the text inputs
    data = []

    # Iterate over each file and append rows to the text_input list
    for file_name in file_names:
        try:
            with open(file_name, "r") as file:
                # Read each line from the file and append to the list
                data.extend([line.strip() for line in file if line.strip()])
        except FileNotFoundError:
            print(f"File not found: {file_name}")
        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")
            
    # Create a DataFrame from the list of text inputs
    df = pd.DataFrame(data, columns=["input"])
    print(df.head())
    return df

process_files()  # Call the function to process files and print the DataFrame head
