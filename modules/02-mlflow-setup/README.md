# Module 02: Experiment Tracking - Logging Parameters, Metrics, and Artifacts

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Beginner |
| **Prerequisites** | Module 01 completed, MLflow tracking server running |

---

## Learning Objectives

By the end of this module, you will be able to:

- Log hyperparameters, metrics, and artifacts to MLflow runs
- Use step-based metric logging to track training progress over epochs
- Log custom artifacts including plots, data samples, and config files
- Organize experiments and compare runs in the MLflow UI
- Use MLflow autologging for common ML frameworks
- Tag runs for filtering and organization

---

## Concepts

### The Anatomy of an MLflow Run

Every MLflow run consists of four types of tracked data:

| Data Type | What It Stores | Examples |
|---|---|---|
| **Parameters** | Input configuration values (immutable per run) | learning_rate=0.01, n_estimators=100, batch_size=32 |
| **Metrics** | Numeric output values (can have multiple steps) | accuracy=0.94, loss=0.23, f1_score=0.91 |
| **Artifacts** | Files and directories | model.pkl, confusion_matrix.png, predictions.csv |
| **Tags** | Key-value metadata strings | developer="alice", sprint="23", gpu="A100" |

### Parameters vs Metrics vs Tags

```
Parameters  = INPUT  configuration  (set once, never changes)
Metrics     = OUTPUT measurements   (can be logged at multiple steps)
Tags        = META   information    (searchable key-value strings)
Artifacts   = FILES  produced       (models, plots, data)
```

### Experiment Organization

Structure your experiments by project or objective, not by individual run:

```
Experiment: "fraud-detection-v2"
  ├── Run: rf_n100_d5       (Random Forest, 100 trees, depth 5)
  ├── Run: rf_n200_d10      (Random Forest, 200 trees, depth 10)
  ├── Run: xgb_lr01_d7      (XGBoost, lr=0.01, depth 7)
  └── Run: lgbm_lr005_d6    (LightGBM, lr=0.05, depth 6)
```

---

## Hands-On Lab

### Exercise 1: Logging Parameters and Metrics

**Goal:** Track a complete sklearn training pipeline with MLflow.

```python
# exercise1_logging_basics.py
import mlflow
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("02-logging-basics")

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameters to try
configs = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 100, "max_depth": None},
]

for config in configs:
    with mlflow.start_run(
        run_name=f"rf_n{config['n_estimators']}_d{config['max_depth']}"
    ):
        # ---- Log Parameters ----
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", config["n_estimators"])
        mlflow.log_param("max_depth", config["max_depth"])
        mlflow.log_param("dataset", "iris")
        mlflow.log_param("test_size", 0.2)

        # ---- Train ----
        model = RandomForestClassifier(**config, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ---- Log Metrics ----
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average="weighted"))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average="weighted"))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="weighted"))

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())

        print(f"Config: {config} -> Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

Run it and open the MLflow UI. Click on the experiment, then use the comparison view to see which configuration performed best.

### Exercise 2: Step-Based Metric Logging (Training Curves)

**Goal:** Log metrics at each training epoch so you can visualize learning curves.

```python
# exercise2_step_metrics.py
import mlflow
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("02-training-curves")

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="mlp-training-curve"):
    mlflow.log_param("model_type", "MLPClassifier")
    mlflow.log_param("hidden_layers", "(64, 32)")
    mlflow.log_param("max_iter", 100)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1,          # We control the loop
        warm_start=True,     # Continue training from previous state
        random_state=42,
    )

    for epoch in range(1, 101):
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        loss = model.loss_

        # Log metrics at each step -- creates a training curve in the UI
        mlflow.log_metrics(
            {
                "train_accuracy": train_score,
                "test_accuracy": test_score,
                "loss": loss,
            },
            step=epoch,
        )

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={loss:.4f}, train_acc={train_score:.4f}, test_acc={test_score:.4f}")

    # Log final metrics without a step
    mlflow.log_metric("final_train_accuracy", train_score)
    mlflow.log_metric("final_test_accuracy", test_score)
```

In the MLflow UI, click on the run and switch to the "Metrics" tab. You will see interactive charts of the training curves.

### Exercise 3: Logging Artifacts (Plots, Files, Directories)

**Goal:** Attach files to your runs for full reproducibility.

```python
# exercise3_artifacts.py
import json
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("02-artifact-logging")

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="gradient-boosting-with-artifacts"):
    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_param("n_estimators", 100)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_metric("accuracy", float(np.mean(y_test == y_pred)))

    # ---- Artifact 1: Confusion matrix plot ----
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    # ---- Artifact 2: Classification report as text ----
    report = classification_report(y_test, y_pred, target_names=wine.target_names)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    # ---- Artifact 3: Feature importance as JSON ----
    importance_dict = dict(zip(wine.feature_names, model.feature_importances_.tolist()))
    with open("feature_importance.json", "w") as f:
        json.dump(importance_dict, f, indent=2)
    mlflow.log_artifact("feature_importance.json")

    # ---- Artifact 4: Log an entire directory ----
    os.makedirs("data_sample", exist_ok=True)
    pd.DataFrame(X_test[:5], columns=wine.feature_names).to_csv("data_sample/test_sample.csv", index=False)
    pd.DataFrame({"actual": y_test[:5], "predicted": y_pred[:5]}).to_csv("data_sample/predictions.csv", index=False)
    mlflow.log_artifacts("data_sample", artifact_path="data_samples")

    print("All artifacts logged. Check the Artifacts tab in the MLflow UI.")
```

### Exercise 4: MLflow Autologging

**Goal:** Let MLflow automatically log parameters, metrics, and models.

```python
# exercise4_autolog.py
import mlflow
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("02-autologging")

# Enable autologging for sklearn -- this is all you need!
mlflow.sklearn.autolog()

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Just train normally -- MLflow captures everything automatically
model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"R2 Score: {score:.4f}")
print("Check the MLflow UI -- autolog captured params, metrics, and the model!")
```

Autologging captured:
- All constructor parameters (n_estimators, max_depth, etc.)
- Training metrics (training_score, etc.)
- The serialized model artifact
- Model signature and input example

### Exercise 5: Tags and Run Organization

**Goal:** Use tags to organize and search across runs.

```python
# exercise5_tags.py
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("02-tagged-runs")

datasets = ["iris", "wine", "diabetes"]
models = ["random_forest", "gradient_boosting"]

for dataset in datasets:
    for model_type in models:
        with mlflow.start_run(run_name=f"{model_type}_{dataset}"):
            # Tags for organization
            mlflow.set_tag("dataset", dataset)
            mlflow.set_tag("model_family", model_type)
            mlflow.set_tag("developer", "student")
            mlflow.set_tag("sprint", "week-2")
            mlflow.set_tag("priority", "high" if dataset == "wine" else "normal")

            # Simulated metrics
            import numpy as np
            mlflow.log_metric("accuracy", np.random.uniform(0.85, 0.98))
            mlflow.log_metric("f1_score", np.random.uniform(0.83, 0.97))

# Now search by tags in the UI or via the API:
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(
    experiment_ids=[mlflow.get_experiment_by_name("02-tagged-runs").experiment_id],
    filter_string="tags.dataset = 'wine' AND tags.model_family = 'random_forest'",
    order_by=["metrics.accuracy DESC"],
)

print(f"\nFound {len(runs)} wine + random_forest runs:")
for run in runs:
    print(f"  Run {run.info.run_id[:8]}: accuracy={run.data.metrics['accuracy']:.4f}")
```

---

## API Reference Quick Guide

```python
# Parameters (set once per run)
mlflow.log_param("key", "value")
mlflow.log_params({"key1": "val1", "key2": "val2"})

# Metrics (can have multiple steps)
mlflow.log_metric("accuracy", 0.95)
mlflow.log_metric("loss", 0.05, step=10)
mlflow.log_metrics({"acc": 0.95, "loss": 0.05}, step=10)

# Artifacts (files and directories)
mlflow.log_artifact("path/to/file.png")
mlflow.log_artifact("file.txt", artifact_path="reports")
mlflow.log_artifacts("directory/", artifact_path="data")

# Tags (key-value strings)
mlflow.set_tag("key", "value")
mlflow.set_tags({"key1": "val1", "key2": "val2"})
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Logging a param twice in the same run | `MlflowException: Changing param values is not allowed` | Log each param only once; use metrics for changing values |
| Logging a metric without steps when you need curves | Flat line in the UI chart | Pass `step=epoch` to `log_metric()` |
| Using `log_artifact` on a directory | Error or unexpected behavior | Use `log_artifacts()` (plural) for directories |
| Not closing runs | Runs stuck in "RUNNING" state | Always use `with mlflow.start_run():` context manager |

---

## Self-Check Questions

1. What is the difference between `log_param` and `log_metric`?
2. How do you log training curves that can be visualized in the MLflow UI?
3. What happens if you call `mlflow.log_param("lr", 0.01)` twice in one run?
4. How do you search for runs by tag values using the MLflow API?
5. When would you use autologging vs manual logging?

---

## You Know You Have Completed This Module When...

- [ ] You logged params, metrics, and artifacts in separate exercises
- [ ] You visualized step-based training curves in the MLflow UI
- [ ] You logged plots, text files, and directories as artifacts
- [ ] You used autologging for an sklearn model
- [ ] You searched runs by tags using the Python API
- [ ] Validation script passes: `bash modules/02-mlflow-setup/validation/validate.sh`

---

## Troubleshooting

### Issue: `MlflowException: Changing param values is not allowed`
You tried to log the same parameter key twice. Parameters are immutable once set.
```python
# Wrong: logging same param twice
mlflow.log_param("lr", 0.01)
mlflow.log_param("lr", 0.001)  # This fails

# Right: use different keys or log as metric
mlflow.log_param("lr", 0.01)
mlflow.log_metric("lr_adjusted", 0.001)
```

### Issue: Artifacts not appearing in UI
- Check that the artifact store path is accessible from the server
- For Docker setups, ensure the MinIO bucket was created
- Verify with: `mlflow.get_artifact_uri()`

---

**Next: [Module 03 - Experiment Tracking and Logging -->](../03-experiment-tracking/)**
