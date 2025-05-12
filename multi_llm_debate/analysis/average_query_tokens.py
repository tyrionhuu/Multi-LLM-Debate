from typing import List

from transformers import AutoTokenizer


def calculate_average_token_count(
    text_list: List[str], image_tokens: int = 0, model_name: str = "bert-base-uncased"
):
    """
    Calculate the average token count of a list of strings, with an optional image token count, using a tokenizer.

    Parameters:
    text_list (list of str): List of strings to calculate token count.
    image_tokens (int, optional): The number of tokens associated with an image. Defaults to 0.
    model_name (str, optional): The name of the pre-trained model for tokenization. Defaults to "bert-base-uncased".

    Returns:
    float: The average number of tokens in the list of strings, including image tokens.
    """
    # Load the tokenizer from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize each string and calculate the number of tokens
    token_counts = [len(tokenizer.tokenize(text)) for text in text_list]

    # Add image tokens if any
    total_tokens = sum(token_counts) + image_tokens

    # Calculate the average token count
    if len(text_list) == 0:  # Avoid division by zero if the list is empty
        return 0

    average_token_count = total_tokens / len(text_list)

    return average_token_count
if __name__ == "__main__":
    from multi_llm_debate.run.big_bench.utils import load_big_bench_dataset

    big_bench_df = load_big_bench_dataset(sample_size=1000)
    big_bench_list = big_bench_df["input"].tolist()
    average_token_count = calculate_average_token_count(big_bench_list)
    print(f"Average token count for BIG_Bench dataset: {average_token_count}")