from typing import List

from sentence_transformers import SentenceTransformer

from .utils import compute_sentence_embedding, kullback_leibler_approximation_distance


def diversity_pruning(
    responses: List[str],
    selected_amount: int,
    model: SentenceTransformer = None,
) -> List[str]:
    """Select a subset of responses that maximizes information entropy.

    The algorithm selects k responses from n candidates that maximize the total
    Kullback-Leibler (KL) divergence between selected responses. This ensures
    maximum diversity in the information content of selected responses.

    Args:
        responses: A list of response strings.
        selected_amount: The number of responses to select (k).
        model: A SentenceTransformer model instance used for encoding.

    Returns:
        A list of selected response strings that maximize information entropy.
    """
    if model is None:
        raise ValueError(
            "A SentenceTransformer model must be provided for diversity pruning."
        )

    if len(responses) <= selected_amount:
        return responses

    # Compute embeddings for all responses
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

    return [responses[i] for i in selected_indices]
