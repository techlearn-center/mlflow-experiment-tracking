"""
Train a scikit-learn model with full MLflow experiment tracking.

This script demonstrates:
- Creating/setting MLflow experiments
- Logging hyperparameters, metrics, and artifacts
- Logging the trained model with input signature
- Comparing runs in the MLflow UI

Usage:
    python -m src.training.train_sklearn
    python -m src.training.train_sklearn --experiment "wine-quality" --max-depth 10
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def load_and_prepare_data(test_size: float = 0.2, random_state: int = 42):
    """Load the Wine dataset and split into train/test sets."""
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, wine.target_names


def plot_confusion_matrix(y_true, y_pred, class_names, output_path: str):
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_feature_importance(model, feature_names, output_path: str):
    """Generate and save a feature importance bar chart."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices], align="center")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title("Feature Importance (Random Forest)")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def train_and_log(
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    experiment_name: str = "sklearn-wine-classifier",
    tracking_uri: str = None,
):
    """Train a Random Forest classifier and log everything to MLflow."""

    # Configure MLflow
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )

    mlflow.set_experiment(experiment_name)

    # Load data
    X_train, X_test, y_train, y_test, scaler, class_names = load_and_prepare_data()

    with mlflow.start_run(run_name=f"rf_n{n_estimators}_d{max_depth}") as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Experiment: {experiment_name}")

        # ---- Log Parameters ----
        params = {
            "model_type": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "max_depth": max_depth if max_depth else "None",
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "test_size": 0.2,
            "random_state": 42,
            "scaler": "StandardScaler",
            "n_features": X_train.shape[1],
            "n_train_samples": X_train.shape[0],
            "n_test_samples": X_test.shape[0],
        }
        mlflow.log_params(params)

        # ---- Train Model ----
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # ---- Evaluate ----
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # Test metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average="weighted")
        test_recall = recall_score(y_test, y_pred, average="weighted")
        test_f1 = f1_score(y_test, y_pred, average="weighted")

        # Train metrics (to check for overfitting)
        train_accuracy = accuracy_score(y_train, y_pred_train)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        # ---- Log Metrics ----
        metrics = {
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "train_accuracy": train_accuracy,
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std(),
            "overfit_gap": train_accuracy - test_accuracy,
        }
        mlflow.log_metrics(metrics)

        print(f"\nTest Accuracy:  {test_accuracy:.4f}")
        print(f"Test F1 Score:  {test_f1:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"CV Accuracy:    {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # ---- Log Artifacts ----
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        # Confusion matrix plot
        cm_path = plot_confusion_matrix(
            y_test, y_pred, class_names,
            str(artifacts_dir / "confusion_matrix.png"),
        )
        mlflow.log_artifact(cm_path)

        # Feature importance plot
        fi_path = plot_feature_importance(
            model, X_train.columns.tolist(),
            str(artifacts_dir / "feature_importance.png"),
        )
        mlflow.log_artifact(fi_path)

        # Classification report as text
        report = classification_report(y_test, y_pred, target_names=class_names)
        report_path = str(artifacts_dir / "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # ---- Log Model with Signature ----
        from mlflow.models.signature import infer_signature

        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:3],
            registered_model_name=f"{experiment_name}",
        )

        # ---- Log Tags ----
        mlflow.set_tags({
            "framework": "scikit-learn",
            "dataset": "wine",
            "task": "classification",
            "developer": "mlflow-training",
        })

        print(f"\nRun logged successfully. View at: {mlflow.get_tracking_uri()}")
        return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description="Train sklearn model with MLflow")
    parser.add_argument("--experiment", type=str, default="sklearn-wine-classifier")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--tracking-uri", type=str, default=None)
    args = parser.parse_args()

    run_id = train_and_log(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        experiment_name=args.experiment,
        tracking_uri=args.tracking_uri,
    )
    print(f"\nCompleted. Run ID: {run_id}")


if __name__ == "__main__":
    main()
