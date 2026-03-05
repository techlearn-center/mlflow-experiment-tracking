"""
Train an XGBoost model with hyperparameter tuning logged to MLflow.

This script demonstrates:
- XGBoost training with MLflow autologging
- Manual hyperparameter search with each trial logged as a nested run
- Early stopping with metrics logged at each boosting round
- Comparing hyperparameter configurations in the MLflow UI

Usage:
    python -m src.training.train_xgboost
    python -m src.training.train_xgboost --n-trials 20 --experiment "xgboost-tuning"
"""

import argparse
import json
import os
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def load_and_prepare_data(test_size: float = 0.2, random_state: int = 42):
    """Load the California Housing dataset."""
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="median_house_value")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, housing.feature_names


def plot_residuals(y_true, y_pred, output_path: str):
    """Generate residual plot."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0].axhline(y=0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted")

    # Residual distribution
    axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Residual Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def train_single_model(
    X_train, X_test, y_train, y_test,
    params: dict,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
):
    """Train a single XGBoost model and return metrics."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dtest, "eval")],
        evals_result=evals_result,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    y_pred = model.predict(dtest)
    y_pred_train = model.predict(dtrain)

    metrics = {
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_r2": float(r2_score(y_test, y_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "best_iteration": int(model.best_iteration),
        "n_trees": int(model.best_iteration + 1),
    }

    return model, metrics, evals_result


def hyperparameter_search(
    n_trials: int = 10,
    experiment_name: str = "xgboost-housing-tuning",
    tracking_uri: str = None,
):
    """Run a hyperparameter search, logging each trial as a nested MLflow run."""

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )

    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()

    # Hyperparameter grid
    param_grid = {
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    # Generate random combinations
    rng = np.random.RandomState(42)
    all_combos = list(product(
        param_grid["max_depth"],
        param_grid["learning_rate"],
        param_grid["subsample"],
        param_grid["colsample_bytree"],
        param_grid["min_child_weight"],
    ))
    selected_indices = rng.choice(len(all_combos), size=min(n_trials, len(all_combos)), replace=False)
    selected_combos = [all_combos[i] for i in selected_indices]

    best_run_id = None
    best_rmse = float("inf")
    results = []

    # Parent run to group all trials
    with mlflow.start_run(run_name="hyperparameter_search") as parent_run:
        mlflow.log_params({
            "search_type": "random",
            "n_trials": n_trials,
            "dataset": "california_housing",
            "n_train_samples": X_train.shape[0],
            "n_test_samples": X_test.shape[0],
        })

        for trial_idx, combo in enumerate(selected_combos):
            max_depth, lr, subsample, colsample, min_child = combo

            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": int(max_depth),
                "learning_rate": float(lr),
                "subsample": float(subsample),
                "colsample_bytree": float(colsample),
                "min_child_weight": int(min_child),
                "seed": 42,
            }

            run_name = f"trial_{trial_idx:03d}_d{max_depth}_lr{lr}"

            # Each trial is a nested run
            with mlflow.start_run(run_name=run_name, nested=True) as child_run:
                mlflow.log_params(params)

                model, metrics, evals_result = train_single_model(
                    X_train, X_test, y_train, y_test, params
                )

                mlflow.log_metrics(metrics)

                # Log training curves as step metrics
                for step, (train_rmse, eval_rmse) in enumerate(zip(
                    evals_result["train"]["rmse"],
                    evals_result["eval"]["rmse"],
                )):
                    mlflow.log_metrics(
                        {"step_train_rmse": train_rmse, "step_eval_rmse": eval_rmse},
                        step=step,
                    )

                # Log the XGBoost model
                mlflow.xgboost.log_model(model, artifact_path="model")

                print(
                    f"Trial {trial_idx + 1}/{len(selected_combos)} | "
                    f"RMSE: {metrics['test_rmse']:.4f} | "
                    f"R2: {metrics['test_r2']:.4f} | "
                    f"depth={max_depth}, lr={lr}"
                )

                # Track best
                if metrics["test_rmse"] < best_rmse:
                    best_rmse = metrics["test_rmse"]
                    best_run_id = child_run.info.run_id

                results.append({
                    "trial": trial_idx,
                    "run_id": child_run.info.run_id,
                    **params,
                    **metrics,
                })

        # Log best trial info to parent run
        mlflow.log_metrics({"best_test_rmse": best_rmse})
        mlflow.set_tag("best_child_run_id", best_run_id)

        # Save results summary as artifact
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        results_df = pd.DataFrame(results)
        results_path = str(artifacts_dir / "tuning_results.csv")
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

        # Generate and log residual plot for the best model
        best_result = results_df.loc[results_df["test_rmse"].idxmin()]
        best_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": int(best_result["max_depth"]),
            "learning_rate": float(best_result["learning_rate"]),
            "subsample": float(best_result["subsample"]),
            "colsample_bytree": float(best_result["colsample_bytree"]),
            "min_child_weight": int(best_result["min_child_weight"]),
            "seed": 42,
        }
        best_model, _, _ = train_single_model(
            X_train, X_test, y_train, y_test, best_params
        )
        dtest = xgb.DMatrix(X_test)
        y_pred_best = best_model.predict(dtest)
        residual_path = plot_residuals(
            y_test.values, y_pred_best,
            str(artifacts_dir / "best_model_residuals.png"),
        )
        mlflow.log_artifact(residual_path)

        # Log search config
        config_path = str(artifacts_dir / "search_config.json")
        with open(config_path, "w") as f:
            json.dump(param_grid, f, indent=2)
        mlflow.log_artifact(config_path)

        print(f"\nBest RMSE: {best_rmse:.4f} (Run ID: {best_run_id})")
        print(f"Parent Run ID: {parent_run.info.run_id}")

    return best_run_id, best_rmse


def main():
    parser = argparse.ArgumentParser(description="XGBoost training with MLflow")
    parser.add_argument("--experiment", type=str, default="xgboost-housing-tuning")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--tracking-uri", type=str, default=None)
    args = parser.parse_args()

    best_run_id, best_rmse = hyperparameter_search(
        n_trials=args.n_trials,
        experiment_name=args.experiment,
        tracking_uri=args.tracking_uri,
    )
    print(f"\nCompleted. Best Run: {best_run_id} (RMSE: {best_rmse:.4f})")


if __name__ == "__main__":
    main()
