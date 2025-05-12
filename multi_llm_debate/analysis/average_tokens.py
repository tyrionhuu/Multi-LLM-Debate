from typing import List, Any
import concurrent.futures
import tiktoken
import json
from pathlib import Path

def tokenize_text(
    text: str, encoder: Any, max_length: int
) -> int:
    """
    Tokenize a single text string and return the token count.
    
    Parameters:
    text (str): The text to tokenize
    encoder (Any): The tiktoken encoder
    max_length (int): Maximum token length
    
    Returns:
    int: Number of tokens in the text
    """
    tokens = encoder.encode(text)
    truncated_tokens = tokens[:max_length]
    return len(truncated_tokens)

def calculate_average_token_count(
    text_list: List[str], image_tokens: int = 0, model_name: str = "o200k_base"
):
    """
    Calculate the average token count of a list of strings, with an optional image token count, using tiktoken.
    Default model is "gpt-4o".

    Parameters:
    text_list (list of str): List of strings to calculate token count.
    image_tokens (int, optional): The number of tokens associated with an image. Defaults to 0.
    model_name (str, optional): The name of the model for tokenization (default is "gpt-3.5-turbo").

    Returns:
    float: The average number of tokens in the list of strings, including image tokens.
    """
    # Initialize the tokenizer for the specified model
    encoder = tiktoken.get_encoding(model_name)

    # Get the model's max token length
    max_length = (
        4096  # GPT-3 models like "gpt-3.5-turbo" have a max token length of 4096
    )

    # Use ThreadPoolExecutor for parallel tokenization
    token_counts = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(tokenize_text, text, encoder, max_length): text 
            for text in text_list
        }
        for future in concurrent.futures.as_completed(futures):
            token_counts.append(future.result())

    # Add image tokens if any
    total_tokens = sum(token_counts) + image_tokens

    # Calculate the average token count
    if len(text_list) == 0:  # Avoid division by zero if the list is empty
        return 0

    average_token_count = total_tokens / len(text_list)

    return average_token_count

def _load_responses_from_json(json_path: str) -> List[str]:
    """
    Load responses from a JSON file and return them as a list of strings.

    Parameters:
    json_path (str): Path to the JSON file.

    Returns:
    list of str: List of responses loaded from the JSON file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract responses from the JSON data
    responses = []
    for item in data:
        if isinstance(item, dict) and 'response' in item:
            responses.append(item['response'])
    
    return responses

def _load_responses_from_round_dir(round_dir: str) -> List[str]:
    """
    Load responses from a round directory and return them as a list of strings.

    Parameters:
    round_dir (str): Path to the round directory.

    Returns:
    list of str: List of responses loaded from the round directory.
    """
    responses = []
    json_files = list(Path(round_dir).glob("*.json"))
    
    # Process JSON files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(_load_responses_from_json, str(json_file)): json_file 
            for json_file in json_files
        }
        for future in concurrent.futures.as_completed(future_to_file):
            responses.extend(future.result())
            
    return responses

def load_responses_from_model_dir(model_dir: str) -> List[str]:
    """
    Load responses from a model directory and return them as a list of strings.
    Gets responses from all subdirectories within the model directory.

    Parameters:
    model_dir (str): Path to the model directory.

    Returns:
    list of str: List of responses loaded from the model directory.
    """
    responses = []
    dir_paths = [d for d in Path(model_dir).iterdir() if d.is_dir()]
    
    # Process directories in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_dir = {
            executor.submit(_load_responses_from_round_dir, str(dir_path)): dir_path 
            for dir_path in dir_paths
        }
        for future in concurrent.futures.as_completed(future_to_dir):
            responses.extend(future.result())
            
    return responses

def calculate_average_token_count_from_model_dir(
    model_dir: str, image_tokens: int = 0, model_name: str = "o200k_base"
) -> float:
    """
    Calculate the average token count of responses from a model directory.

    Parameters:
    model_dir (str): Path to the model directory.
    image_tokens (int, optional): The number of tokens associated with an image. Defaults to 0.
    model_name (str, optional): The name of the model for tokenization (default is "gpt-3.5-turbo").

    Returns:
    float: The average number of tokens in the responses from the model directory.
    """
    responses = load_responses_from_model_dir(model_dir)
    print(f"Loaded {len(responses)} responses from model directory {model_dir}")
    return calculate_average_token_count(responses, image_tokens, model_name)


if __name__ == "__main__":
    model_dir = "data/big_bench/gemma-3-4b-it(7)"
    average_token_count = calculate_average_token_count_from_model_dir(model_dir)
    print(f"Average token count for model directory {model_dir}: {average_token_count}")
    
    model_dir = "data/judge_bench/gemma-3-4b-it(7)"
    average_token_count = calculate_average_token_count_from_model_dir(model_dir)
    print(f"Average token count for model directory {model_dir}: {average_token_count}")
    
    model_dir = "data/llm_bar/gemma-3-4b-it(7)"
    average_token_count = calculate_average_token_count_from_model_dir(model_dir)
    print(f"Average token count for model directory {model_dir}: {average_token_count}")
    
    model_dir = "data/mllm_judge_pair/gemma-3-4b-it(7)"
    average_token_count = calculate_average_token_count_from_model_dir(model_dir)
    print(f"Average token count for model directory {model_dir}: {average_token_count}")
    
    model_dir = "data/judge_anything_pair/gemma-3-4b-it(7)"
    average_token_count = calculate_average_token_count_from_model_dir(model_dir)
    print(f"Average token count for model directory {model_dir}: {average_token_count}")
    
    model_dir = "data/truthful_qa/gemma-3-4b-it(7)"
    average_token_count = calculate_average_token_count_from_model_dir(model_dir)
    print(f"Average token count for model directory {model_dir}: {average_token_count}")
