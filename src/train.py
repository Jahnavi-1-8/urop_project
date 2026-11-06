"""Model Training Module for Apple Leaf Disease Classification.

Trains exactly two machine learning classifiers:
1. Random Forest Classifier
2. Support Vector Machine (SVM with RBF kernel and probability calibration)

Adheres strictly to academic ML best practices:
- Stratified train-test splitting (prevents class imbalance distortion)
- Feature scaling (MinMaxScaler) fitted SOLELY on training data (no data leakage)
- Configurable hyperparameters with reproducible random seeds
- Persistence of trained models, scaler, and label encoder via joblib
"""

from typing import Dict, Any, Tuple, Optional
import os
import sys
import argparse
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Ensure project root is in sys.path when executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.feature_extraction import extract_features_from_dataset


DEFAULT_FEATURES_PATH = os.path.join("data", "processed", "apple_leaf_features.npz")
DEFAULT_MODELS_DIR = os.path.join("models")
DEFAULT_SPLIT_DATA_PATH = os.path.join("data", "processed", "split_data.npz")
RANDOM_SEED = 42


def load_extracted_data(features_path: str = DEFAULT_FEATURES_PATH) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-extracted features or run feature extraction if not present.

    Args:
        features_path: Path to .npz file containing extracted features.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (features, labels, file_paths)
    """
    if not os.path.exists(features_path):
        print(f"[INFO] Processed features not found at {features_path}. Running feature extraction...")
        extract_features_from_dataset(output_path=features_path)

    data = np.load(features_path, allow_pickle=True)
    X = data["features"]
    y = data["labels"]
    file_paths = data["file_paths"]
    print(f"[INFO] Successfully loaded features: {X.shape}, labels: {y.shape}")
    return X, y, file_paths


def prepare_data_splits(
    X: np.ndarray,
    y: np.ndarray,
    file_paths: np.ndarray,
    test_size: float = 0.20,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Perform stratified split and fit feature scaler exclusively on training set.

    Avoids data leakage by fitting MinMaxScaler strictly on X_train.

    Args:
        X: Feature matrix.
        y: Class label strings.
        file_paths: Corresponding image paths.
        test_size: Fraction of samples for testing (default 0.20 -> 80/20 split).
        random_state: Reproducibility seed.

    Returns:
        Dict[str, Any]: Split datasets, fitted scaler, and label encoder.
    """
    # 1. Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 2. Stratified Train-Test Split
    (
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_enc,
        y_test_enc,
        paths_train,
        paths_test,
    ) = train_test_split(
        X,
        y,
        y_encoded,
        file_paths,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    print(f"\n========================================================")
    print(f" DATASET SPLIT & FEATURE SCALING")
    print(f"========================================================")
    print(f"Total samples: {len(y)} | Train samples: {len(y_train)} ({(1-test_size)*100:.0f}%) | Test samples: {len(y_test)} ({test_size*100:.0f}%)")
    print("Class distribution in Train set:")
    for cls_name, count in zip(*np.unique(y_train, return_counts=True)):
        print(f"  - {cls_name}: {count}")
    print("Class distribution in Test set:")
    for cls_name, count in zip(*np.unique(y_test, return_counts=True)):
        print(f"  - {cls_name}: {count}")

    # 3. Fit Scaler ONLY on Training Data (prevents test data leakage)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = np.clip(scaler.transform(X_test), 0.0, 1.0)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_encoded": y_train_enc,
        "y_test_encoded": y_test_enc,
        "paths_train": paths_train,
        "paths_test": paths_test,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    random_state: int = RANDOM_SEED,
) -> RandomForestClassifier:
    """Train Random Forest Classifier.

    Args:
        X_train: Training feature matrix.
        y_train: Encoded target labels.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples required to split a node.
        random_state: Reproducibility seed.

    Returns:
        RandomForestClassifier: Fitted model.
    """
    print(f"\n[TRAINING] Random Forest Classifier (n_estimators={n_estimators}, max_depth={max_depth})...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    train_acc = rf.score(X_train, y_train)
    print(f"[STATUS] Random Forest training complete. Training accuracy: {train_acc * 100:.2f}%")
    return rf


def train_svm(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    c_param: float = 10.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    random_state: int = RANDOM_SEED,
) -> SVC:
    """Train Support Vector Machine Classifier with calibrated probabilities.

    Args:
        X_train_scaled: Scaled training feature matrix (critical for SVM).
        y_train: Encoded target labels.
        c_param: Regularization parameter C.
        kernel: Kernel type ('rbf', 'linear', etc.).
        gamma: Kernel coefficient ('scale' or 'auto').
        random_state: Reproducibility seed.

    Returns:
        SVC: Fitted Support Vector Classifier.
    """
    print(f"\n[TRAINING] Support Vector Machine (kernel='{kernel}', C={c_param}, gamma='{gamma}', probability=True)...")
    svm = SVC(
        C=c_param,
        kernel=kernel,
        gamma=gamma,
        probability=True,
        class_weight="balanced",
        random_state=random_state,
    )
    svm.fit(X_train_scaled, y_train)
    train_acc = svm.score(X_train_scaled, y_train)
    print(f"[STATUS] SVM training complete. Training accuracy: {train_acc * 100:.2f}%")
    return svm


def run_training_pipeline(
    features_path: str = DEFAULT_FEATURES_PATH,
    models_dir: str = DEFAULT_MODELS_DIR,
    split_data_path: str = DEFAULT_SPLIT_DATA_PATH,
    rf_n_estimators: int = 100,
    svm_c: float = 10.0,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Execute complete training workflow and save models and artifacts."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(split_data_path), exist_ok=True)

    # 1. Load data
    X, y, file_paths = load_extracted_data(features_path)

    # 2. Split and scale
    splits = prepare_data_splits(X, y, file_paths, test_size=0.20, random_state=random_state)

    # 3. Train Random Forest
    rf_model = train_random_forest(
        splits["X_train_scaled"],
        splits["y_train_encoded"],
        n_estimators=rf_n_estimators,
        random_state=random_state,
    )

    # 4. Train SVM
    svm_model = train_svm(
        splits["X_train_scaled"],
        splits["y_train_encoded"],
        c_param=svm_c,
        kernel="rbf",
        random_state=random_state,
    )

    # 5. Persist models, scaler, and label encoder
    rf_path = os.path.join(models_dir, "random_forest.joblib")
    svm_path = os.path.join(models_dir, "svm.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    le_path = os.path.join(models_dir, "label_encoder.joblib")

    joblib.dump(rf_model, rf_path)
    joblib.dump(svm_model, svm_path)
    joblib.dump(splits["scaler"], scaler_path)
    joblib.dump(splits["label_encoder"], le_path)

    # 6. Save split test/train data for independent evaluation
    np.savez_compressed(
        split_data_path,
        X_train=splits["X_train"],
        X_test=splits["X_test"],
        X_train_scaled=splits["X_train_scaled"],
        X_test_scaled=splits["X_test_scaled"],
        y_train=splits["y_train"],
        y_test=splits["y_test"],
        y_train_encoded=splits["y_train_encoded"],
        y_test_encoded=splits["y_test_encoded"],
        paths_train=splits["paths_train"],
        paths_test=splits["paths_test"],
        classes=splits["label_encoder"].classes_,
    )

    print(f"\n========================================================")
    print(f" TRAINING PIPELINE COMPLETE")
    print(f"========================================================")
    print(f"Artifacts saved:")
    print(f" - Random Forest Model: {rf_path}")
    print(f" - SVM Model:           {svm_path}")
    print(f" - Feature Scaler:      {scaler_path}")
    print(f" - Label Encoder:       {le_path}")
    print(f" - Train/Test Split:    {split_data_path}")

    return {
        "rf_model": rf_model,
        "svm_model": svm_model,
        "scaler": splits["scaler"],
        "label_encoder": splits["label_encoder"],
        "test_data_path": split_data_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest and SVM on Apple Leaf Dataset")
    parser.add_argument("--features", default=DEFAULT_FEATURES_PATH, help="Path to features npz")
    parser.add_argument("--models_dir", default=DEFAULT_MODELS_DIR, help="Directory to save models")
    parser.add_argument("--rf_estimators", type=int, default=100, help="Number of trees in Random Forest")
    parser.add_argument("--svm_c", type=float, default=10.0, help="C parameter for SVM")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_training_pipeline(
        features_path=args.features,
        models_dir=args.models_dir,
        rf_n_estimators=args.rf_estimators,
        svm_c=args.svm_c,
        random_state=args.seed,
    )
