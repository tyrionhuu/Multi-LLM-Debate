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
    
    # Get the model's max token length
    max_length = tokenizer.model_max_length

    # Tokenize each string and calculate the number of tokens
    token_counts = []
    for text in text_list:
        # Tokenize with truncation and padding to ensure it fits the max length
        tokens = tokenizer.tokenize(text)
        token_counts.append(min(len(tokens), max_length))  # Ensure we don't exceed max_length

    # Add image tokens if any
    total_tokens = sum(token_counts) + image_tokens

    # Calculate the average token count
    if len(text_list) == 0:  # Avoid division by zero if the list is empty
        return 0

    average_token_count = total_tokens / len(text_list)

    return average_token_count

if __name__ == "__main__":
    from multi_llm_debate.run.big_bench.utils import load_big_bench_dataset
    from multi_llm_debate.run.judge_bench.utils import load_judge_bench_dataset
    from multi_llm_debate.run.llm_bar.utils import load_llm_bar_dataset
    from multi_llm_debate.run.truthful_qa.utils import load_truthful_qa_dataset

    # big_bench_df = load_big_bench_dataset(sample_size=1000)
    # big_bench_list = big_bench_df["input"].tolist()
    # average_token_count = calculate_average_token_count(big_bench_list)
    # print(f"Average token count for BIG_Bench dataset: {average_token_count}")
    
    # judge_bench_df = load_judge_bench_dataset()
    # judge_bench_df["merged_input"] = judge_bench_df["question"] + " " + judge_bench_df["response_A"] + " " + judge_bench_df["response_B"]
    # judge_bench_list = judge_bench_df["merged_input"].tolist()
    # average_token_count = calculate_average_token_count(judge_bench_list)
    # print(f"Average token count for Judge_Bench dataset: {average_token_count}")
    
    # llm_bar_df = load_llm_bar_dataset()
    # llm_bar_df["merged_input"] = llm_bar_df["question"] + " " + llm_bar_df["response_1"] + " " + llm_bar_df["response_2"]
    # llm_bar_list = llm_bar_df["merged_input"].tolist()
    # average_token_count = calculate_average_token_count(llm_bar_list)
    # print(f"Average token count for LLM_Bar dataset: {average_token_count}")
    
    truthful_qa_df = load_truthful_qa_dataset()
    truthful_qa_df["merged_input"] = truthful_qa_df["question"] + " " + truthful_qa_df["answer_A"] + " " + truthful_qa_df["answer_B"] + " " + truthful_qa_df["answer_C"]
    truthful_qa_list = truthful_qa_df["merged_input"].tolist()
    average_token_count = calculate_average_token_count(truthful_qa_list)
    print(f"Average token count for Truthful_QA dataset: {average_token_count}")