from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import compute_sentence_embedding, kullback_leibler_approximation_distance


def quality_pruning(
    responses: List[str],
    task: str,
    selected_amount: int,
    model: SentenceTransformer = None,
) -> List[str]:
    """Select a subset of responses that are most similar to the task (maximizing quality).

    The algorithm selects k responses from n candidates that minimize the KL divergence
    between the task (x) and the response (zi). This ensures that the selected responses
    are most relevant to the task.

    Args:
        responses: A list of response strings.
        task: The task string to compare the responses to.
        selected_amount: The number of responses to select (k).
        model: A SentenceTransformer model instance used for encoding.

    Returns:
        A list of selected response strings that are most similar to the task.
    """
    if model is None:
        raise ValueError(
            "A SentenceTransformer model must be provided for quality pruning."
        )

    if len(responses) <= selected_amount:
        return responses

    # Compute the embedding for the task
    task_embedding = compute_sentence_embedding(model, task)

    # Compute the embeddings for all responses
    embeddings = [compute_sentence_embedding(model, response) for response in responses]

    # Compute the cosine distance (KL approximation) between the task and each response
    distances = [
        kullback_leibler_approximation_distance(task_embedding, embedding)
        for embedding in embeddings
    ]

    # Select the indices of the k responses that are closest to the task (minimize distance)
    selected_indices = np.argsort(distances)[:selected_amount]

    # Return the selected responses based on the indices
    return [responses[i] for i in selected_indices]
