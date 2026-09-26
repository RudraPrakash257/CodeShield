"""Train the ML model."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from src.ml.model import train_model, save_model
from src.ml.features import get_feature_names
from src.extractors.pdf import extract_text_from_pdf
from src.extractors.docx import extract_text_from_docx
from src.extractors.text import extract_text_from_file
from src.ml.features import extract_pairwise_features
import pandas as pd
import numpy as np


def load_documents():
    """Load documents from data directory."""
    docs = []
    data_dir = Path("data/documents")

    if not data_dir.exists():
        print("No data/documents directory found")
        return docs

    # Also check handwriting
    hw_dir = Path("data/handwriting")
    if hw_dir.exists():
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for f in hw_dir.glob(ext):
                docs.append({"filename": f.name, "path": str(f), "type": "handwriting"})

    for f in data_dir.glob("*.pdf"):
        try:
            doc = extract_text_from_pdf(str(f))
            doc["path"] = str(f)
            doc["type"] = "pdf"
            docs.append(doc)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    for f in data_dir.glob("*.docx"):
        try:
            doc = extract_text_from_docx(str(f))
            doc["path"] = str(f)
            doc["type"] = "docx"
            docs.append(doc)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    for f in data_dir.glob("*.txt"):
        try:
            doc = extract_text_from_file(str(f))
            doc["path"] = str(f)
            doc["type"] = "txt"
            docs.append(doc)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    return docs


def main():
    """Train the model."""
    print("Loading documents...")
    documents = load_documents()

    if len(documents) < 2:
        print("Need at least 2 documents for training")
        print(f"Found: {len(documents)}")
        return

    print(f"Loaded {len(documents)} documents")

    # Load labels
    pairs_path = Path("data/pairs.csv")
    hw_pairs_path = Path("data/handwriting_pairs.csv")

    if not pairs_path.exists() and not hw_pairs_path.exists():
        print("No training data found!")
        print("Please create data/pairs.csv or data/handwriting_pairs.csv")
        print("Format: doc_a,doc_b,label")
        return

    # Load label files
    try:
        pairs_df = pd.read_csv(pairs_path)
    except:
        pairs_df = pd.DataFrame(columns=["doc_a", "doc_b", "label"])

    try:
        hw_pairs_df = pd.read_csv(hw_pairs_path)
    except:
        hw_pairs_df = pd.DataFrame(columns=["image_a", "image_b", "label"])

    total_pairs = len(pairs_df) + len(hw_pairs_df)

    print(f"Training pairs: {total_pairs}")
    print(f"  - Document pairs: {len(pairs_df)}")
    print(f"  - Handwriting pairs: {len(hw_pairs_df)}")

    if total_pairs == 0:
        print("No labeled pairs found!")
        return

    # Create document lookup
    doc_lookup = {doc["filename"]: doc for doc in documents}

    # Extract features
    print("\nExtracting features...")
    X = []
    y = []
    feature_names = get_feature_names()

    # Process document pairs
    for _, row in pairs_df.iterrows():
        doc_a = doc_lookup.get(row["doc_a"])
        doc_b = doc_lookup.get(row["doc_b"])

        if doc_a and doc_b:
            features = extract_pairwise_features(doc_a, doc_b)

            feature_vector = []
            for name in feature_names:
                value = features.get(name, 0) or 0
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
                value = features.get(name, 0) or 0
                feature_vector.append(value)

            X.append(feature_vector)
            y.append(row["label"])

    X = np.array(X)
    y = np.array(y)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Train model
    print("\nTraining model...")
    model_data = train_model(X, y, model_type='auto')

    print(f"\nModel type: {model_data['model_type']}")
    print(f"Accuracy: {model_data['metrics']['accuracy']:.2%}")
    print(f"Precision: {model_data['metrics']['precision']:.2%}")
    print(f"Recall: {model_data['metrics']['recall']:.2%}")
    print(f"F1: {model_data['metrics']['f1']:.2%}")

    # Save model
    print("\nSaving model...")
    model_path, metadata_path = save_model(model_data, feature_names)

    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()