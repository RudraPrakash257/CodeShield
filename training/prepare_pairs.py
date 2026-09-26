"""Prepare training data for ML model."""
import pandas as pd
import numpy as np
from pathlib import Path
from src.ml.features import extract_pairwise_features, get_feature_names


def prepare_training_data(
    documents: list,
    labels_path: str = "data/pairs.csv",
    handwriting_labels_path: str = "data/handwriting_pairs.csv"
) -> tuple:
    """
    Prepare feature matrix and labels from training data.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels array (n_samples,)
    """
    # Load labels
    try:
        pairs_df = pd.read_csv(labels_path)
    except FileNotFoundError:
        pairs_df = pd.DataFrame(columns=["doc_a", "doc_b", "label"])

    # Load handwriting labels if exists
    try:
        hw_pairs_df = pd.read_csv(handwriting_labels_path)
    except FileNotFoundError:
        hw_pairs_df = pd.DataFrame(columns=["image_a", "image_b", "label"])

    # Create document lookup
    doc_lookup = {doc["filename"]: doc for doc in documents}

    # Extract features for each pair
    X = []
    y = []

    feature_names = get_feature_names()

    # Process document pairs
    for _, row in pairs_df.iterrows():
        doc_a = doc_lookup.get(row["doc_a"])
        doc_b = doc_lookup.get(row["doc_b"])

        if doc_a and doc_b:
            features = extract_pairwise_features(doc_a, doc_b)

            # Build feature vector
            feature_vector = []
            for name in feature_names:
                value = features.get(name, 0)
                if value is None:
                    value = 0
                feature_vector.append(value)

            X.append(feature_vector)
            y.append(row["label"])

    # Process handwriting pairs
    for _, row in hw_pairs_df.iterrows():
        doc_a = doc_lookup.get(row["image_a"])
        doc_b = doc_lookup.get(row["image_b"])

        if doc_a and doc_b:
            features = extract_pairwise_features(doc_a, doc_b)

            feature_vector = []
            for name in feature_names:
                value = features.get(name, 0)
                if value is None:
                    value = 0
                feature_vector.append(value)

            X.append(feature_vector)
            y.append(row["label"])

    if not X:
        return None, None

    return np.array(X), np.array(y)