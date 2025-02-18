import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

def compute_sentence_embedding(model: SentenceTransformer, sentence: str) -> np.ndarray:
    """Compute the embedding vector for a given sentence using a SentenceTransformer model.

    Args:
        model: A SentenceTransformer model instance used for encoding.
        sentence: The input text to be encoded into an embedding vector.

    Returns:
        A numpy array containing the sentence embedding vector.
    """
    return model.encode([sentence])[0]

def kullback_leibler_approximation_distance(
    embedding1: np.ndarray, embedding2: np.ndarray
) -> float:
    """Compute the Kullback-Leibler divergence between two embeddings.

    Args:
        embedding1: The first embedding vector.
        embedding2: The second embedding vector.

    Returns:
        The Kullback-Leibler divergence between the two embeddings.
    """
    return cosine(embedding1, embedding2)