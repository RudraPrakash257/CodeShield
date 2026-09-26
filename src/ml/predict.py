"""ML model prediction."""
import numpy as np
from src.ml.features import get_feature_names


def predict_similarity(model, features: dict, metadata: dict) -> float:
    """
    Predict similarity score for a document pair.

    Args:
        model: Trained model
        features: Feature dict from extract_pairwise_features
        metadata: Model metadata with feature names

    Returns:
        Similarity score (0-1)
    """
    # Get feature vector in correct order
    feature_names = metadata.get('feature_names', get_feature_names())

    feature_vector = []
    for name in feature_names:
        value = features.get(name, 0)
        # Handle None values
        if value is None:
            value = 0.0
        feature_vector.append(value)

    # Convert to numpy array
    X = np.array(feature_vector).reshape(1, -1)

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    # Predict probability of similarity
    try:
        proba = model.predict_proba(X)[0, 1]
        return float(proba)
    except:
        # Fallback to binary prediction
        pred = model.predict(X)[0]
        return float(pred)


def calculate_heuristic_score(features: dict) -> float:
    """
    Calculate heuristic similarity score when ML model is unavailable.

    Weighted combination of key features.
    """
    lexical = features.get('lexical_similarity', 0) or 0
    semantic = features.get('semantic_similarity', 0) or 0
    structural = features.get('structural_similarity', 0) or 0

    # Weighted average
    overall = 0.30 * lexical + 0.50 * semantic + 0.20 * structural

    return float(overall)
