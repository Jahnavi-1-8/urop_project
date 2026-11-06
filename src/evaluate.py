"""Model Evaluation & Comparison Module for Apple Leaf Disease Classification.

Evaluates trained Random Forest and Support Vector Machine (SVM) models on held-out test data.
Computes:
- Overall Accuracy
- Macro and Weighted Precision
- Macro and Weighted Recall
- Macro and Weighted F1-Score
- Class-level Classification Reports
- Confusion Matrices

Generates academic visualizations saved to results/:
1. results/class_distribution.png
2. results/confusion_matrix_rf.png
3. results/confusion_matrix_svm.png
4. results/model_comparison.png
5. results/model_comparison.csv
6. results/per_class_metrics.csv
"""

from typing import Dict, Any, List
import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Ensure project root is in sys.path when executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.feature_extraction import CLASS_MAP


DEFAULT_MODELS_DIR = os.path.join("models")
DEFAULT_SPLIT_DATA_PATH = os.path.join("data", "processed", "split_data.npz")
DEFAULT_RESULTS_DIR = os.path.join("results")


def set_plot_style():
    """Configure modern, academic plot aesthetic."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
    })


def load_test_artifacts(
    models_dir: str = DEFAULT_MODELS_DIR,
    split_data_path: str = DEFAULT_SPLIT_DATA_PATH,
) -> Dict[str, Any]:
    """Load trained models, scaler, label encoder, and split test dataset."""
    rf_path = os.path.join(models_dir, "random_forest.joblib")
    svm_path = os.path.join(models_dir, "svm.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    le_path = os.path.join(models_dir, "label_encoder.joblib")

    if not all(os.path.exists(p) for p in [rf_path, svm_path, scaler_path, le_path, split_data_path]):
        raise FileNotFoundError(
            "Required model or data artifacts missing. Please run `python src/train.py` first."
        )

    rf_model = joblib.load(rf_path)
    svm_model = joblib.load(svm_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(le_path)

    data = np.load(split_data_path, allow_pickle=True)
    X_test_scaled = data["X_test_scaled"]
    y_test_encoded = data["y_test_encoded"]
    y_test = data["y_test"]
    y_train = data["y_train"]
    classes = data["classes"]

    return {
        "rf_model": rf_model,
        "svm_model": svm_model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "X_test_scaled": X_test_scaled,
        "y_test_encoded": y_test_encoded,
        "y_test": y_test,
        "y_train": y_train,
        "classes": classes,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
    """Compute comprehensive academic classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    return {
        "model": model_name,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
    }


def plot_class_distribution(
    y_train: np.ndarray,
    y_test: np.ndarray,
    classes: List[str],
    output_path: str,
):
    """Plot dataset split distribution across all four apple classes."""
    set_plot_style()
    readable_classes = [CLASS_MAP.get(c, c) for c in classes]

    train_counts = [np.sum(y_train == c) for c in classes]
    test_counts = [np.sum(y_test == c) for c in classes]

    df_dist = pd.DataFrame({
        "Class": readable_classes * 2,
        "Count": train_counts + test_counts,
        "Split": ["Train (80%)"] * len(classes) + ["Test (20%)"] * len(classes),
    })

    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.barplot(
        data=df_dist,
        x="Class",
        y="Count",
        hue="Split",
        palette=["#2b5c8f", "#d95f02"],
    )

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{int(h)}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.title("PlantVillage Apple Leaf Class Distribution (Train vs Test)", pad=15, fontweight="bold")
    plt.xlabel("Disease Category")
    plt.ylabel("Number of Samples")
    plt.legend(title="Dataset Partition")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] Class distribution plot: {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    model_name: str,
    output_path: str,
):
    """Plot normalized and annotated confusion matrix."""
    set_plot_style()
    readable_classes = [CLASS_MAP.get(c, c) for c in classes]

    plt.figure(figsize=(8, 6.5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=readable_classes,
        yticklabels=readable_classes,
        cbar=True,
        linewidths=1,
        linecolor="#e0e0e0",
    )

    plt.title(f"Confusion Matrix — {model_name}", pad=15, fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] {model_name} confusion matrix: {output_path}")


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: str):
    """Plot grouped comparison bar chart for RF vs SVM."""
    set_plot_style()

    plot_df = comparison_df.melt(
        id_vars=["model"],
        value_vars=["accuracy", "precision", "recall", "f1_score"],
        var_name="Metric",
        value_name="Score",
    )

    metric_names = {
        "accuracy": "Accuracy",
        "precision": "Precision (Macro)",
        "recall": "Recall (Macro)",
        "f1_score": "F1-Score (Macro)",
    }
    plot_df["Metric"] = plot_df["Metric"].map(metric_names)

    plt.figure(figsize=(9, 6), dpi=300)
    ax = sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="model",
        palette=["#1f77b4", "#ff7f0e"],
    )

    for p in ax.patches:
        val = p.get_height()
        ax.annotate(
            f"{val:.3f}",
            (p.get_x() + p.get_width() / 2.0, val),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            xytext=(0, 3),
            textcoords="offset points",
        )

    plt.title("Model Performance Comparison: Random Forest vs SVM", pad=15, fontweight="bold")
    plt.ylabel("Score (0.0 – 1.0)")
    plt.xlabel("Evaluation Metric")
    plt.ylim(0.0, 1.05)
    plt.legend(title="Classifier", loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] Model comparison plot: {output_path}")


def evaluate_models(
    models_dir: str = DEFAULT_MODELS_DIR,
    split_data_path: str = DEFAULT_SPLIT_DATA_PATH,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """Run full evaluation suite for both models and export results."""
    os.makedirs(results_dir, exist_ok=True)
    artifacts = load_test_artifacts(models_dir, split_data_path)

    X_test_scaled = artifacts["X_test_scaled"]
    y_test_encoded = artifacts["y_test_encoded"]
    y_test = artifacts["y_test"]
    y_train = artifacts["y_train"]
    classes = artifacts["classes"]
    readable_classes = [CLASS_MAP.get(c, c) for c in classes]

    # Predictions
    rf_pred = artifacts["rf_model"].predict(X_test_scaled)
    svm_pred = artifacts["svm_model"].predict(X_test_scaled)

    # Compute metrics
    rf_metrics = compute_metrics(y_test_encoded, rf_pred, "Random Forest")
    svm_metrics = compute_metrics(y_test_encoded, svm_pred, "SVM")

    # Build primary comparison dataframe
    comparison_records = [
        {
            "model": "Random Forest",
            "accuracy": rf_metrics["accuracy"],
            "precision": rf_metrics["precision_macro"],
            "recall": rf_metrics["recall_macro"],
            "f1_score": rf_metrics["f1_macro"],
            "precision_weighted": rf_metrics["precision_weighted"],
            "recall_weighted": rf_metrics["recall_weighted"],
            "f1_score_weighted": rf_metrics["f1_weighted"],
        },
        {
            "model": "SVM",
            "accuracy": svm_metrics["accuracy"],
            "precision": svm_metrics["precision_macro"],
            "recall": svm_metrics["recall_macro"],
            "f1_score": svm_metrics["f1_macro"],
            "precision_weighted": svm_metrics["precision_weighted"],
            "recall_weighted": svm_metrics["recall_weighted"],
            "f1_score_weighted": svm_metrics["f1_weighted"],
        },
    ]
    comparison_df = pd.DataFrame(comparison_records)

    # Save CSV with core columns as requested in prompt: model, accuracy, precision, recall, f1_score
    csv_path = os.path.join(results_dir, "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Comparison CSV: {csv_path}")

    # Build per-class classification reports
    rf_rep = classification_report(y_test_encoded, rf_pred, target_names=readable_classes, output_dict=True)
    svm_rep = classification_report(y_test_encoded, svm_pred, target_names=readable_classes, output_dict=True)

    per_class_records = []
    for cls in readable_classes:
        per_class_records.append({
            "Class": cls,
            "RF_Precision": rf_rep[cls]["precision"],
            "RF_Recall": rf_rep[cls]["recall"],
            "RF_F1": rf_rep[cls]["f1-score"],
            "SVM_Precision": svm_rep[cls]["precision"],
            "SVM_Recall": svm_rep[cls]["recall"],
            "SVM_F1": svm_rep[cls]["f1-score"],
            "Support": rf_rep[cls]["support"],
        })
    per_class_df = pd.DataFrame(per_class_records)
    per_class_csv = os.path.join(results_dir, "per_class_metrics.csv")
    per_class_df.to_csv(per_class_csv, index=False)

    # Visualizations
    dist_plot_path = os.path.join(results_dir, "class_distribution.png")
    plot_class_distribution(y_train, y_test, classes, dist_plot_path)

    rf_cm_path = os.path.join(results_dir, "confusion_matrix_rf.png")
    plot_confusion_matrix(rf_metrics["confusion_matrix"], classes, "Random Forest", rf_cm_path)

    svm_cm_path = os.path.join(results_dir, "confusion_matrix_svm.png")
    plot_confusion_matrix(svm_metrics["confusion_matrix"], classes, "SVM", svm_cm_path)

    comp_plot_path = os.path.join(results_dir, "model_comparison.png")
    plot_model_comparison(comparison_df, comp_plot_path)

    # Console display table
    print("\n" + "=" * 65)
    print(" MODEL PERFORMANCE COMPARISON TABLE (HELD-OUT TEST SET, N=635)")
    print("=" * 65)
    format_df = comparison_df[["model", "accuracy", "precision", "recall", "f1_score"]].copy()
    format_df["accuracy"] = format_df["accuracy"].map(lambda x: f"{x:.4f} ({x*100:.2f}%)")
    format_df["precision"] = format_df["precision"].map(lambda x: f"{x:.4f}")
    format_df["recall"] = format_df["recall"].map(lambda x: f"{x:.4f}")
    format_df["f1_score"] = format_df["f1_score"].map(lambda x: f"{x:.4f}")
    print(format_df.to_string(index=False))

    print("\n" + "=" * 65)
    print(" PER-CLASS PERFORMANCE BREAKDOWN")
    print("=" * 65)
    print(per_class_df.to_string(index=False))

    # Save detailed text report
    report_path = os.path.join(results_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("APPLE LEAF DISEASE CLASSIFICATION — EVALUATION REPORT\n")
        f.write("=====================================================\n\n")
        f.write("1. MODEL COMPARISON SUMMARY\n")
        f.write(comparison_df.to_string(index=False) + "\n\n")
        f.write("2. RANDOM FOREST CLASSIFICATION REPORT\n")
        f.write(classification_report(y_test_encoded, rf_pred, target_names=readable_classes) + "\n\n")
        f.write("3. SUPPORT VECTOR MACHINE CLASSIFICATION REPORT\n")
        f.write(classification_report(y_test_encoded, svm_pred, target_names=readable_classes) + "\n\n")
        f.write("4. RANDOM FOREST CONFUSION MATRIX\n")
        f.write(np.array2string(rf_metrics["confusion_matrix"]) + "\n\n")
        f.write("5. SUPPORT VECTOR MACHINE CONFUSION MATRIX\n")
        f.write(np.array2string(svm_metrics["confusion_matrix"]) + "\n")
    print(f"\n[SAVED] Full evaluation text report: {report_path}")

    return comparison_df


if __name__ == "__main__":
    evaluate_models()
