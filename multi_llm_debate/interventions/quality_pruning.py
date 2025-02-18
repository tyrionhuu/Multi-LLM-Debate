from typing import List

from sentence_transformers import SentenceTransformer

from .utils import compute_sentence_embedding, kullback_leibler_approximation_distance


def quality_pruning(
    responses: List[str],
    task: str,
    selected_amount: int,
    model: SentenceTransformer = None,
) -> List[str]:
    """Select a subset of responses that maximize quality and diversity.

    The algorithm selects k responses from n candidates that maximize the total
    Kullback-Leibler (KL) divergence between selected responses, while also
    maximizing the quality of the responses. Quality is determined by the cosine
    similarity between the response and the task prompt.

    Args:
        responses: A list of response strings.
        task: The task prompt string.
        selected_amount: The number of responses to select (k).
        model: A SentenceTransformer model instance used for encoding.

    Returns:
        A list of selected response strings that maximize quality and diversity.
    """
    if model is None:
        raise ValueError(
            "A SentenceTransformer model must be provided for quality pruning."
        )

    if len(responses) <= selected_amount:
        return responses

    # Compute embeddings for task prompt and all responses
    task_embedding = compute_sentence_embedding(model, task)
    response_embeddings = [compute_sentence_embedding(model, response) for response in responses]

    # Start with the highest quality response
    selected_indices = [max(
        range(len(responses)),
        key=lambda i: 1 - kullback_leibler_approximation_distance(task_embedding, response_embeddings[i])
    )]

    # Iteratively select responses that maximize total KL divergence and quality
    while len(selected_indices) < selected_amount:
        max_total_kl = float("-inf")
        next_index = -1

        # For each candidate response
        for i in range(len(responses)):
            if i in selected_indices:
                continue

            # Calculate total KL divergence if we add this response
            total_kl = sum(
                kullback_leibler_approximation_distance(response_embeddings[i], response_embeddings[j])
                for j in selected_indices
            )

            # Calculate quality of this response
            quality = 1 - kullback_leibler_approximation_distance(task_embedding, response_embeddings[i])

            # Combine quality and diversity scores
            score = total_kl + quality

            if score > max_total_kl:
                max_total_kl = score
                next_index = i

        selected_indices.append(next_index)

    return [responses[i] for i in selected_indices]