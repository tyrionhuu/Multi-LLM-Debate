from scipy.stats import entropy
import numpy as np


def kullback_leibler_divergence(p: list[float], q: list[float]) -> float:
    """
    Calculate the Kullback-Leibler divergence between two probability distributions.

    Args:
        p: The first probability distribution.
        q: The second probability distribution.

    Returns:
        The Kullback-Leibler divergence between p and q.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    return entropy(p, q)