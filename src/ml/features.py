"""Feature extraction for ML model."""
from src.similarity.lexical import (
    calculate_lexical_similarity,
    calculate_word_overlap,
    calculate_phrase_overlap,
    find_matching_phrases
)
from src.similarity.semantic import calculate_semantic_similarity
from src.similarity.structural import calculate_structural_similarity


def extract_pairwise_features(doc_a: dict, doc_b: dict) -> dict:
    """
    Extract all pairwise features for a document pair.

    Returns:
        dict with all similarity features
    """
    text_a = doc_a.get("text", "")
    text_b = doc_b.get("text", "")

    # Text-based features
    features = {
        "lexical_similarity": calculate_lexical_similarity(text_a, text_b),
        "semantic_similarity": calculate_semantic_similarity(text_a, text_b),
        "structural_similarity": calculate_structural_similarity(doc_a, doc_b),
        "word_overlap": calculate_word_overlap(text_a, text_b),
        "phrase_overlap": calculate_phrase_overlap(text_a, text_b)
    }

    # Find matching phrases for evidence
    features["matching_phrases"] = find_matching_phrases(text_a, text_b)

    return features


def get_feature_names():
    """Return list of feature names for ML model."""
    return [
        "lexical_similarity",
        "semantic_similarity",
        "structural_similarity",
        "word_overlap",
        "phrase_overlap"
    ]
