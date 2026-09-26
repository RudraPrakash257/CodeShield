"""Semantic similarity using sentence transformers."""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Global model instance
_semantic_model = None


def get_semantic_model():
    """Get or create semantic model instance."""
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _semantic_model


def calculate_semantic_similarity(text_a: str, text_b: str) -> float:
    """Calculate semantic similarity using sentence embeddings."""
    if not text_a or not text_b:
        return 0.0

    try:
        model = get_semantic_model()

        # Get embeddings
        embeddings = model.encode([text_a, text_b])

        # Calculate cosine similarity
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]

        return float(similarity)

    except Exception as e:
        return 0.0
