import json
from pathlib import Path
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import compute_sentence_embedding, kullback_leibler_approximation_distance


def quality_pruning(
    responses: List[str],
    input: str,
    selected_amount: int = 5,
    model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2"),
    output_dir: Union[str, Path] = None,
    round_number: int = 0,
) -> List[str]:
    """Select a subset of responses that are most similar to the input (maximizing quality).

    The algorithm selects k responses from n candidates that minimize the KL divergence
    between the input (x) and the response (zi). This ensures that the selected responses
    are most relevant to the input.

    Args:
        responses: A list of response strings.
        input: The input string to compare the responses to.
        selected_amount: The number of responses to select (k).
        model: A SentenceTransformer model instance used for encoding.

    Returns:
        A list of selected response strings that are most similar to the input.
    """
    if model is None:
        raise ValueError(
            "A SentenceTransformer model must be provided for quality pruning."
        )

    if input is None or len(input.strip()) == 0:
        return responses

    if len(responses) <= selected_amount:
        return responses

    # Compute the embedding for the input
    task_embedding = compute_sentence_embedding(model, input)

    # Compute the embeddings for all responses
    embeddings = [compute_sentence_embedding(model, response) for response in responses]

    # Compute the cosine distance (KL approximation) between the input and each response
    distances = [
        kullback_leibler_approximation_distance(task_embedding, embedding)
        for embedding in embeddings
    ]

    # Select the indices of the k responses that are closest to the input (minimize distance)
    selected_indices = np.argsort(distances)[:selected_amount]

    selected_responses = [responses[i] for i in selected_indices]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"debate_round_{round_number}.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "selected_indices": selected_indices.tolist(),
                    "selected_responses": selected_responses,
                    "total_responses": len(responses),
                },
                f,
                indent=2,
            )

    return selected_responses
