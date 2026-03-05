# Module 08: Hyperparameter Tuning with Optuna and MLflow

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 07 completed, `pip install optuna` |

---

## Learning Objectives

By the end of this module, you will be able to:

- Integrate Optuna with MLflow for hyperparameter optimization
- Log each Optuna trial as a nested MLflow run
- Visualize hyperparameter search results in the MLflow UI
- Use different Optuna samplers (TPE, Random, CMA-ES)
- Implement pruning to stop unpromising trials early
- Register the best model from a hyperparameter search

---

## Concepts

### Why Optuna + MLflow?

| Tool | Responsibility |
|---|---|
| **Optuna** | Decides which hyperparameters to try next (Bayesian optimization) |
| **MLflow** | Records every trial's params, metrics, and artifacts |

Together they give you intelligent search **and** a permanent record of every experiment.

### Optuna Key Concepts

| Term | Definition |
|---|---|
| **Study** | A hyperparameter optimization session |
| **Trial** | A single evaluation with a specific set of hyperparameters |
| **Objective** | The function to minimize or maximize |
| **Sampler** | Algorithm for choosing hyperparameters (TPE, Random, CMA-ES) |
| **Pruner** | Early-stopping strategy for unpromising trials |

### How It Fits Together

```
Optuna Study
  ├── Trial 1 ──> MLflow Nested Run (params, metrics, model)
  ├── Trial 2 ──> MLflow Nested Run (params, metrics, model)
  ├── Trial 3 ──> MLflow Nested Run (PRUNED at epoch 5)
  └── Trial N ──> MLflow Nested Run (params, metrics, model)
         │
         v
  Best Trial ──> Register in Model Registry
```

---

## Hands-On Lab

### Exercise 1: Basic Optuna + MLflow Integration

**Goal:** Run Optuna optimization with each trial logged to MLflow.

```python
# exercise1_optuna_basic.py
import mlflow
import optuna
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("08-optuna-basic")

# Load data globally so it is shared across trials
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)


def objective(trial):
    """Optuna objective function -- each call is one trial."""
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    # Log each trial as a nested MLflow run
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": str(max_features),
        })

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )

        # Cross-validation for more robust evaluation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        mean_cv = cv_scores.mean()
        std_cv = cv_scores.std()

        # Fit on full train set and evaluate on test set
        model.fit(X_train, y_train)
        test_accuracy = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_metrics({
            "cv_accuracy_mean": mean_cv,
            "cv_accuracy_std": std_cv,
            "test_accuracy": test_accuracy,
        })

        mlflow.set_tag("trial_number", trial.number)

        print(f"Trial {trial.number}: cv_acc={mean_cv:.4f}, test_acc={test_accuracy:.4f}")

    return mean_cv  # Optuna will maximize this


# Run the study inside a parent MLflow run
with mlflow.start_run(run_name="optuna-rf-study"):
    mlflow.log_param("study_type", "TPE")
    mlflow.log_param("n_trials", 30)
    mlflow.log_param("objective_metric", "cv_accuracy")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    # Log best trial results to parent run
    best = study.best_trial
    mlflow.log_metric("best_cv_accuracy", best.value)
    mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})

    print(f"\nBest trial: {best.number}")
    print(f"Best CV accuracy: {best.value:.4f}")
    print(f"Best params: {best.params}")
```

### Exercise 2: XGBoost Tuning with Pruning

**Goal:** Use Optuna pruning to stop unpromising XGBoost trials early.

```python
# exercise2_xgboost_pruning.py
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("08-xgboost-pruning")

housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "seed": 42,
    }

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(params)

        # Custom callback for Optuna pruning
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "eval-rmse")

        evals_result = {}
        try:
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=500,
                evals=[(dtrain, "train"), (dtest, "eval")],
                evals_result=evals_result,
                early_stopping_rounds=30,
                callbacks=[pruning_callback],
                verbose_eval=False,
            )
        except optuna.TrialPruned:
            # Log that this trial was pruned
            mlflow.set_tag("pruned", "true")
            n_completed = len(evals_result.get("eval", {}).get("rmse", []))
            mlflow.log_metric("rounds_completed", n_completed)
            raise

        # Log step metrics
        for step, (train_rmse, eval_rmse) in enumerate(zip(
            evals_result["train"]["rmse"],
            evals_result["eval"]["rmse"],
        )):
            mlflow.log_metrics({"train_rmse": train_rmse, "eval_rmse": eval_rmse}, step=step)

        y_pred = model.predict(dtest)
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("best_iteration", model.best_iteration)
        mlflow.set_tag("pruned", "false")

        # Log the model for the best trials
        if trial.number < 5 or test_rmse < 0.5:
            mlflow.xgboost.log_model(model, "model")

        print(f"Trial {trial.number}: RMSE={test_rmse:.4f}, rounds={model.best_iteration}")

    return test_rmse


with mlflow.start_run(run_name="xgboost-optuna-pruning"):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )

    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # Log summary
    mlflow.log_metric("best_rmse", study.best_value)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("n_pruned", len(study.get_trials(states=[optuna.trial.TrialState.PRUNED])))
    mlflow.log_metric("n_completed", len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])))

    print(f"\nBest RMSE: {study.best_value:.4f}")
    print(f"Pruned: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}/{len(study.trials)}")
```

### Exercise 3: Visualize Hyperparameter Importance

**Goal:** Generate and log Optuna visualization plots.

```python
# exercise3_visualize.py
import mlflow
import optuna
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)


def log_optuna_visualizations(study, experiment_name="08-visualizations"):
    """Generate Optuna plots and log them as MLflow artifacts."""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="optuna-visualizations"):
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_image("optimization_history.png")
        mlflow.log_artifact("optimization_history.png")

        # Parameter importance
        fig = plot_param_importances(study)
        fig.write_image("param_importances.png")
        mlflow.log_artifact("param_importances.png")

        # Parallel coordinate plot
        fig = plot_parallel_coordinate(study)
        fig.write_image("parallel_coordinate.png")
        mlflow.log_artifact("parallel_coordinate.png")

        # Contour plot for top 2 parameters
        try:
            importances = optuna.importance.get_param_importances(study)
            top_params = list(importances.keys())[:2]
            if len(top_params) == 2:
                fig = plot_contour(study, params=top_params)
                fig.write_image("contour_plot.png")
                mlflow.log_artifact("contour_plot.png")
        except Exception:
            pass

        # Save study results as CSV
        df = study.trials_dataframe()
        df.to_csv("study_results.csv", index=False)
        mlflow.log_artifact("study_results.csv")

        print("Visualizations logged to MLflow artifacts!")


# Use with any completed study:
# log_optuna_visualizations(study)
```

### Exercise 4: Register the Best Model

**Goal:** After tuning, train the best model and register it.

```python
# exercise4_register_best.py
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("08-register-tuned-model")

# Best params from Optuna study (replace with your actual results)
best_params = {
    "n_estimators": 280,
    "max_depth": 12,
    "min_samples_split": 3,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="best-tuned-model"):
    mlflow.log_params(best_params)
    mlflow.set_tag("tuning_method", "optuna_tpe")
    mlflow.set_tag("tuning_trials", "30")

    model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", accuracy)

    signature = infer_signature(X_test, y_pred)

    # Register the tuned model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=X_test[:3],
        registered_model_name="wine-quality-classifier",
    )

    print(f"Tuned model registered with accuracy: {accuracy:.4f}")
```

---

## Comparing Samplers

| Sampler | Algorithm | Best For |
|---|---|---|
| `TPESampler` | Tree-structured Parzen Estimator | General-purpose, good default |
| `RandomSampler` | Random search | Baseline comparison, high-dimensional spaces |
| `CmaEsSampler` | CMA Evolution Strategy | Continuous parameters, smaller search spaces |
| `GridSampler` | Exhaustive grid | Small parameter grids |

```python
# Try different samplers
study_tpe = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
study_random = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
study_cma = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(seed=42))
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Not using nested runs | Parent run cluttered with trial data | Use `nested=True` in `mlflow.start_run()` |
| Forgetting to handle pruned trials | `optuna.TrialPruned` crashes the run | Catch the exception and re-raise it |
| Too few startup trials for pruner | Good trials pruned prematurely | Set `n_startup_trials >= 5` |
| Not seeding the sampler | Non-reproducible results | Pass `seed` to the sampler |

---

## Self-Check Questions

1. How does Optuna's TPE sampler differ from random search?
2. What is pruning and when should you use it?
3. How do nested MLflow runs help organize hyperparameter search results?
4. What is the purpose of `n_startup_trials` in a pruner?
5. How would you compare the effectiveness of different Optuna samplers?

---

## You Know You Have Completed This Module When...

- [ ] You ran an Optuna study with MLflow logging for each trial
- [ ] You used pruning to stop unpromising XGBoost trials early
- [ ] You generated and logged visualization plots
- [ ] You registered the best tuned model in the MLflow registry
- [ ] You can explain TPE vs random sampling
- [ ] Validation script passes: `bash modules/08-ab-testing/validation/validate.sh`

---

**Next: [Module 09 - Monitoring Integration -->](../09-monitoring-integration/)**
