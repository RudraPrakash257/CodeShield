"""ML model training and management."""
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def train_model(X, y, model_type='auto', random_state=42):
    """
    Train similarity classification model.

    Args:
        X: Feature matrix
        y: Labels (1=similar, 0=dissimilar)
        model_type: 'xgboost', 'random_forest', or 'auto'
        random_state: Random seed

    Returns:
        dict with model, metrics, and metadata
    """
    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Choose model
    if model_type == 'auto':
        model_type = 'xgboost' if XGBOOST_AVAILABLE else 'random_forest'

    if model_type == 'xgboost' and XGBOOST_AVAILABLE:
        model = xgb.XGBClassifier(
            max_depth=5,
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric='logloss'
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
        model_type = 'random_forest'

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)) if len(np.unique(y_test)) > 1 else None,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    return {
        'model': model,
        'model_type': model_type,
        'metrics': metrics,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }


def save_model(model_data: dict, feature_names: list, model_dir: str = 'models'):
    """Save model and metadata."""
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    # Save model
    model_path = model_dir / 'codeshield_model.pkl'
    joblib.dump(model_data['model'], model_path)

    # Save metadata
    metadata = {
        'model_type': model_data['model_type'],
        'feature_names': feature_names,
        'metrics': model_data['metrics'],
        'train_size': model_data['train_size'],
        'test_size': model_data['test_size'],
        'training_date': datetime.now().isoformat(),
        'model_version': '1.0'
    }

    metadata_path = model_dir / 'codeshield_metadata.pkl'
    joblib.dump(metadata, metadata_path)

    return model_path, metadata_path


def load_model(model_dir: str = 'models'):
    """Load model and metadata."""
    model_dir = Path(model_dir)

    model_path = model_dir / 'codeshield_model.pkl'
    metadata_path = model_dir / 'codeshield_metadata.pkl'

    if not model_path.exists() or not metadata_path.exists():
        return None, None

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)

    return model, metadata
