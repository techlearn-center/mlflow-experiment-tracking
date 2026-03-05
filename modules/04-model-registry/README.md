# Module 04: MLflow Projects - Reproducible ML Code Packaging

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 03 completed, Docker installed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Create an MLproject file to define reproducible ML workflows
- Configure conda and Docker environments for MLflow Projects
- Run MLflow Projects locally and from Git repositories
- Pass parameters to project entry points
- Chain multiple project steps into multi-step workflows
- Understand how Projects enable reproducibility across teams

---

## Concepts

### What is an MLflow Project?

An MLflow Project is a convention for packaging ML code so anyone can reproduce your results. A project is simply a directory (or Git repo) with an `MLproject` file that specifies:

1. **Name** -- Project identifier
2. **Environment** -- How to set up dependencies (conda, Docker, or system)
3. **Entry points** -- Commands to run with typed parameters

### The MLproject File

```yaml
name: wine-classifier

# Option 1: Conda environment
conda_env: conda.yaml

# Option 2: Docker environment
# docker_env:
#   image: my-ml-image:latest
#   volumes: ["/data:/data"]

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 5}
      test_size: {type: float, default: 0.2}
    command: "python train.py --n-estimators {n_estimators} --max-depth {max_depth} --test-size {test_size}"

  evaluate:
    parameters:
      run_id: {type: string}
    command: "python evaluate.py --run-id {run_id}"
```

### Why Projects Matter

| Without Projects | With Projects |
|---|---|
| "It works on my machine" | Reproducible on any machine |
| Manual dependency management | Environment auto-created |
| Undocumented parameters | Typed, documented parameters with defaults |
| One-off scripts | Reusable, shareable workflows |

---

## Hands-On Lab

### Exercise 1: Create Your First MLflow Project

**Goal:** Build a complete ML project with an MLproject file.

**Step 1:** Create the project directory structure.

```bash
mkdir -p mlflow-wine-project
cd mlflow-wine-project
```

**Step 2:** Create `conda.yaml`.

```yaml
# conda.yaml
name: wine-classifier-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - mlflow>=2.12.0
      - scikit-learn>=1.4.0
      - pandas>=2.2.0
      - numpy>=1.26.0
      - matplotlib>=3.8.0
```

**Step 3:** Create `MLproject`.

```yaml
# MLproject
name: wine-classifier

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 5}
      learning_rate: {type: float, default: 0.1}
      data_path: {type: string, default: "default"}
    command: >
      python train.py
      --n-estimators {n_estimators}
      --max-depth {max_depth}
      --learning-rate {learning_rate}
      --data-path {data_path}

  evaluate:
    parameters:
      model_uri: {type: string}
    command: "python evaluate.py --model-uri {model_uri}"

  preprocess:
    command: "python preprocess.py"
```

**Step 4:** Create `train.py`.

```python
# train.py
import argparse

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--data-path", type=str, default="default")
    args = parser.parse_args()

    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
        })

        model = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=42,
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})

        signature = infer_signature(X_test_s, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)

        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()
```

**Step 5:** Run the project.

```bash
# Run with default parameters
mlflow run . --experiment-name "04-mlflow-projects"

# Run with custom parameters
mlflow run . -P n_estimators=200 -P max_depth=7 -P learning_rate=0.05

# Run a specific entry point
mlflow run . -e preprocess
```

### Exercise 2: Run Projects from Git

**Goal:** Run an MLflow Project directly from a Git repository.

```bash
# Run any public MLflow project from GitHub
mlflow run https://github.com/mlflow/mlflow-example -P alpha=0.5

# Run with specific branch or commit
mlflow run https://github.com/mlflow/mlflow-example --version main -P alpha=0.3
```

```python
# Run projects programmatically
import mlflow

submitted_run = mlflow.projects.run(
    uri="https://github.com/mlflow/mlflow-example",
    parameters={"alpha": 0.5},
    experiment_name="04-git-projects",
    synchronous=True,
)

print(f"Run ID: {submitted_run.run_id}")
print(f"Status: {submitted_run.get_status()}")
```

### Exercise 3: Docker-Based Projects

**Goal:** Use Docker instead of conda for perfectly reproducible environments.

```dockerfile
# Dockerfile.project
FROM python:3.11-slim

RUN pip install mlflow scikit-learn pandas numpy matplotlib

WORKDIR /app
COPY . /app
```

```yaml
# MLproject (Docker version)
name: wine-classifier-docker

docker_env:
  image: wine-classifier:latest
  volumes:
    - "${PWD}/data:/app/data"
  environment:
    - ["MLFLOW_TRACKING_URI", "http://host.docker.internal:5000"]

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 5}
    command: "python train.py --n-estimators {n_estimators} --max-depth {max_depth}"
```

```bash
docker build -f Dockerfile.project -t wine-classifier:latest .
mlflow run . --experiment-name "04-docker-projects"
```

### Exercise 4: Multi-Step Workflows

**Goal:** Chain multiple project steps together.

```python
# workflow.py
"""
Multi-step workflow:
1. Preprocess data
2. Train model with different hyperparameters
3. Evaluate the best model
"""
import mlflow

EXPERIMENT_NAME = "04-multi-step-workflow"


def run_workflow():
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="full-pipeline") as parent_run:
        # Step 1: Preprocess
        print("Step 1: Preprocessing...")
        preprocess_run = mlflow.projects.run(
            uri=".",
            entry_point="preprocess",
            experiment_name=EXPERIMENT_NAME,
            synchronous=True,
        )

        # Step 2: Train with different configs
        configs = [
            {"n_estimators": "50", "max_depth": "3", "learning_rate": "0.1"},
            {"n_estimators": "100", "max_depth": "5", "learning_rate": "0.05"},
            {"n_estimators": "200", "max_depth": "7", "learning_rate": "0.01"},
        ]

        best_run_id = None
        best_metric = 0

        for i, config in enumerate(configs):
            print(f"Step 2.{i + 1}: Training with {config}...")
            train_run = mlflow.projects.run(
                uri=".",
                entry_point="main",
                parameters=config,
                experiment_name=EXPERIMENT_NAME,
                synchronous=True,
            )

            client = mlflow.tracking.MlflowClient()
            run_data = client.get_run(train_run.run_id)
            accuracy = run_data.data.metrics.get("accuracy", 0)

            if accuracy > best_metric:
                best_metric = accuracy
                best_run_id = train_run.run_id

        # Step 3: Log the best result
        mlflow.log_param("best_run_id", best_run_id)
        mlflow.log_metric("best_accuracy", best_metric)

        print(f"Pipeline complete. Best accuracy: {best_metric:.4f}")


if __name__ == "__main__":
    run_workflow()
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Missing `MLproject` file | `mlflow.exceptions.ExecutionException` | Create MLproject in the project root |
| Conda env creation fails | Dependency resolution errors | Pin versions in conda.yaml; consider Docker |
| Wrong parameter types | `TypeError` during project run | Match types in MLproject (int, float, string) |
| Docker not finding tracking server | Connection refused from container | Use `host.docker.internal` or a network bridge |

---

## Self-Check Questions

1. What are the three environment options for MLflow Projects?
2. How do you pass parameters when running a project from the CLI?
3. What is the benefit of running a project from a Git URI?
4. When would you choose Docker over conda for a project environment?
5. How do multi-step workflows help with complex ML pipelines?

---

## You Know You Have Completed This Module When...

- [ ] You created an MLproject file with parameters and entry points
- [ ] You ran a project locally with custom parameters
- [ ] You ran a project from a Git repository
- [ ] You understand conda vs Docker trade-offs
- [ ] You built a multi-step workflow
- [ ] Validation script passes: `bash modules/04-model-registry/validation/validate.sh`

---

**Next: [Module 05 - Model Packaging and Signatures -->](../05-model-packaging/)**
