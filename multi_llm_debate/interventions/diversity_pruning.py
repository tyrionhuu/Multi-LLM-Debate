import json
import random
from pathlib import Path
from typing import Callable, List, Union

from sentence_transformers import SentenceTransformer

from .utils import compute_sentence_embedding, kullback_leibler_approximation_distance


def diversity_pruning_by_embedding(
    responses: List[str],
    selected_amount: int = 5,
    model: SentenceTransformer = None,
    output_dir: Union[str, Path] = None,
    round_number: int = 0,
    **kwargs,
) -> List[str]:
    """Select a subset of responses that maximizes information entropy.

    The algorithm selects k responses from n candidates that maximize the total
    Kullback-Leibler (KL) divergence between selected responses. This ensures
    maximum diversity in the information content of selected responses.

    Args:
        responses: A list of response strings.
        selected_amount: The number of responses to select (k).
        model: A SentenceTransformer model instance used for encoding.
               Defaults to 'all-MiniLM-L6-v2'.
        output_dir: Directory path to save intermediate results (if needed).
        round_number: The current round number for saving intermediate results.
        **kwargs: Additional keyword arguments.

    Returns:
        A list of selected response strings that maximize information entropy.
    """
    if len(responses) < selected_amount:
        return responses

    # Compute embeddings for all responses
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = [compute_sentence_embedding(model, response) for response in responses]

    # Start with the first response
    selected_indices = [0]

    # Iteratively select responses that maximize total KL divergence
    while len(selected_indices) < selected_amount:
        max_total_kl = float("-inf")
        next_index = -1

        # For each candidate response
        for i in range(len(embeddings)):
            if i in selected_indices:
                continue

            # Calculate total KL divergence if we add this response
            total_kl = sum(
                kullback_leibler_approximation_distance(embeddings[i], embeddings[j])
                for j in selected_indices
            )

            if total_kl > max_total_kl:
                max_total_kl = total_kl
                next_index = i

        selected_indices.append(next_index)

    selected_responses = [responses[i] for i in selected_indices]

    # Save responses if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"debate_round_{round_number}.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "selected_indices": selected_indices,
                    "selected_responses": selected_responses,
                    "total_responses": len(responses),
                },
                f,
                indent=2,
            )

    return selected_responses


def diversity_pruning_by_answer(
    responses: List[str],
    selected_amount: int = 5,
    extract_func: Callable = None,
    random_seed: int = 42,
    output_dir: Union[str, Path] = None,
    round_number: int = 0,
    **kwargs,
) -> List[str]:
    """
    Select a subset of responses that maximizes information entropy on final answers.

    This function selects responses with different final answers. If there are fewer
    unique answers than the requested amount, additional responses will be included
    to meet the requested amount (which may result in duplicate answer types).

    Args:
        responses: A list of response strings.
        selected_amount: The number of responses to select (k).
        extract_func: A function that extracts the final answer from the response.
        random_seed: Seed for random selection when filling remaining slots.
        output_dir: Directory path to save intermediate results (if needed).
        round_number: The current round number for saving intermediate results.
        **kwargs: Additional keyword arguments.

    Returns:
        A list of selected response strings with exactly selected_amount items
        (unless the input has fewer total responses).

    Raises:
        ValueError: If no extraction function is provided or any response lacks a
            valid final answer.
    """
    if len(responses) < selected_amount:
        return responses

    if extract_func is None:
        raise ValueError(
            "An extraction function must be provided for diversity pruning by answer."
        )

    random.seed(random_seed)

    # Extract final answers from responses
    extracted_answers = []
    for response in responses:
        extracted_response = extract_func(response)
        if extracted_response is None:
            raise ValueError(
                "All responses must have a valid final answer for diversity pruning by answer."
            )
        # Convert list extracted answers to str for uniqueness comparison
        if isinstance(extracted_response, list):
            extracted_response_for_set = str(extracted_response)
        else:
            extracted_response_for_set = extracted_response
        extracted_answers.append((extracted_response, extracted_response_for_set))

    # Select responses with different answers
    selected_indices = []
    seen_answers = set()

    # First pass: collect responses with unique answers
    for i, (answer, answer_for_set) in enumerate(extracted_answers):
        if (
            answer_for_set not in seen_answers
            and len(selected_indices) < selected_amount
        ):
            selected_indices.append(i)
            seen_answers.add(answer_for_set)

    # Second pass: if we don't have enough responses, add more even if answers repeat
    if len(selected_indices) < selected_amount:
        remaining_indices = [
            i for i in range(len(responses)) if i not in selected_indices
        ]
        # Shuffle the remaining indices to introduce randomness
        random.shuffle(remaining_indices)
        # Add additional responses up to the requested amount
        selected_indices.extend(
            remaining_indices[: selected_amount - len(selected_indices)]
        )

    selected_responses = [responses[i] for i in selected_indices]

    # Save responses if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"debate_round_{round_number}.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "selected_indices": selected_indices,
                    "selected_responses": selected_responses,
                    "total_responses": len(responses),
                },
                f,
                indent=2,
            )

    return selected_responses


def main() -> None:
    """Test function for diversity pruning algorithms.

    This function demonstrates the use of both embedding-based and
    answer-based diversity pruning methods on a set of example responses.
    """
    # Example responses for testing
    test_responses = [
        "The answer is 42 because it is the meaning of life.",
        "I believe the answer is 42, as per Douglas Adams.",
        "The solution to this problem is 7 squared, which equals 49.",
        "After careful calculation, I determine the answer is 42.",
        "Based on the evidence, the answer must be 36.",
        "The correct response is 42, without a doubt.",
        "I would argue that 36 is the right answer to this question.",
        "My analysis shows that 49 is the correct value.",
        "The most logical conclusion is that the answer is 36.",
        "According to my calculations, the answer is 49.",
    ]

    # Define a simple extraction function that looks for numbers in the response
    def extract_numerical_answer(response: str) -> str:
        """Extract numerical answer from a response string.

        Args:
            response: A string containing the full response.

        Returns:
            The first number found in the response as a string.
        """
        import re

        numbers = re.findall(r"\d+", response)
        return numbers[0] if numbers else None

    print("Original responses:")
    for i, response in enumerate(test_responses):
        print(f"{i+1}. {response}")
    print("\n" + "=" * 50 + "\n")

    # Test embedding-based pruning
    print("Testing embedding-based diversity pruning (select 3):")
    embedding_selected = diversity_pruning_by_embedding(
        test_responses, selected_amount=3
    )

    for i, response in enumerate(embedding_selected):
        print(f"{i+1}. {response}")
    print("\n" + "=" * 50 + "\n")

    # Test answer-based pruning
    print("Testing answer-based diversity pruning (select 3):")
    answer_selected = diversity_pruning_by_answer(
        test_responses, selected_amount=3, extract_func=extract_numerical_answer
    )

    for i, response in enumerate(answer_selected):
        print(f"{i+1}. {response} (extracted: {extract_numerical_answer(response)})")


if __name__ == "__main__":
    main()
