import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

def compute_sentence_embedding(model: SentenceTransformer, sentence: str) -> np.ndarray:
    return model.encode([sentence])[0]