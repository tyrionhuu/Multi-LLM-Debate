from typing import List

from sentence_transformers import SentenceTransformer

from .utils import kullback_leibler_approximation_distance


def diversity_pruning(
    responses: List[str],
    selected_amount: int,
    model: SentenceTransformer = None,
) -> List[str]:
    """Select a subset of responses that are diverse in content.

    Args:
        responses: A list of response strings.
        selected_amount: The number of responses to select.
        model: A SentenceTransformer model instance used for encoding.

    Returns:
        A list of selected response strings.
    """
    if model is None:
        raise ValueError(
            "A SentenceTransformer model must be provided for diversity pruning."
        )

    if len(responses) <= selected_amount:
        return responses

    embeddings = [model.encode([response])[0] for response in responses]
    selected_indices = [0]
    while len(selected_indices) < selected_amount:
        max_distance = 0
        max_index = -1
        for i, embedding in enumerate(embeddings):
            if i in selected_indices:
                continue
            distance = min(
                kullback_leibler_approximation_distance(embedding, embeddings[j])
                for j in selected_indices
            )
            if distance > max_distance:
                max_distance = distance
                max_index = i
        selected_indices.append(max_index)

    return [responses[i] for i in selected_indices]
