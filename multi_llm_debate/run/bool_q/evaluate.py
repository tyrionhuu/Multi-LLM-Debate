from typing import Dict, List, Union

from ...llm.parsers import extract_bool_answer


def evaluate_bool_responses(
    responses: List[Dict],
    answer: Union[str, bool],
) -> bool:
    """Evaluate the responses from the debate.

    Args:
        responses: List of agent responses from the most recent round of debate.
        answer: The correct answer to the question ("yes"/"no", "true"/"false", or bool).

    Returns:
        bool: True if all responses are the same and match the answer, False otherwise.
    """
    try:
        raw_responses = [response["response"] for response in responses]
        normalized_responses = [
            extract_bool_answer(response) for response in raw_responses
        ]
        answer_string = str(answer).lower()

        # print(f"Normalized answer: {normalized_responses}")
        # Filter out empty responses
        valid_responses = [r for r in normalized_responses if r]
        if not valid_responses:
            # print("Warning: No valid responses found")
            return False

        # Check if all valid responses are the same
        if len(set(valid_responses)) == 1:
            return valid_responses[0] == answer_string
        return False
    except Exception as e:
        print(f"Error evaluating responses: {e}")
        return False


def main() -> None:
    from pathlib import Path

    from ...utils.download_dataset import load_save_dataset_df
    from ..shared.evaluate import (
        evaluate_debate_df,
        evaluate_ensemble_df,
        evaluate_single_llm_df,
    )
    from .utils import process_bool_q_df

    dataset_path = Path("datasets/boolq")

    # Example usage
    response_base_dir = Path("data/bool_q/phi3")
    dataframe = load_save_dataset_df(
        dataset_name="google/boolq",
        dataset_path=dataset_path,
        force_download=False,
    )

    # Process the DataFrame
    processed_dataframe = process_bool_q_df(dataframe)
    # Evaluate the debate responses
    evaluate_debate_df(response_base_dir, processed_dataframe)

    # Evaluate the single LLM responses
    evaluate_single_llm_df(response_base_dir, processed_dataframe)

    # Evaluate the ensemble responses using the shared function
    evaluate_ensemble_df(
        response_base_dir,
        processed_dataframe,
        extract_func=extract_bool_answer,
        evaluation_func=evaluate_bool_responses,
    )


if __name__ == "__main__":
    main()
