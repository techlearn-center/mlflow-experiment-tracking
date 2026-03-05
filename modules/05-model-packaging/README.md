# Module 05: Model Registry - Registration, Versioning, and Stage Transitions

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 04 completed, MLflow server with database backend running |

---

## Learning Objectives

By the end of this module, you will be able to:

- Register models from experiment runs into the MLflow Model Registry
- Manage model versions and understand automatic version incrementing
- Transition models through stages: None, Staging, Production, Archived
- Add descriptions and tags to models and versions
- Compare model versions programmatically
- Build an automated promotion pipeline based on metric thresholds

---

## Concepts

### The Model Registry Lifecycle

```
Training Run ──> Register ──> Staging ──> Production
                    │                        │
                    │                        v
                    v                    Archived
               Version N+1
```

The Model Registry is a centralized store that tracks:
- **Registered Models**: Named entries that group model versions
- **Model Versions**: Immutable snapshots linked to specific training runs
- **Stage Labels**: Lifecycle state (None, Staging, Production, Archived)
- **Descriptions and Tags**: Metadata for governance and documentation

### Stage Transitions

| Stage | Meaning | Who Uses It |
|---|---|---|
| **None** | Just registered, not validated | Data scientists after training |
| **Staging** | Being validated and tested | QA/ML engineers during validation |
| **Production** | Serving live traffic | Serving infrastructure |
| **Archived** | Retired, kept for audit trail | Compliance and rollback |

---

## Hands-On Lab

### Exercise 1: Register a Model from a Training Run

**Goal:** Train a model, log it, then register it in the Model Registry.

```python
# exercise1_register_model.py
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("05-model-registry")

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

MODEL_NAME = "wine-quality-classifier"

with mlflow.start_run(run_name="register-v1") as run:
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    signature = infer_signature(X_test, y_pred)

    # registered_model_name auto-registers the model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        registered_model_name=MODEL_NAME,
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Run ID: {run.info.run_id}")
```

### Exercise 2: Create Multiple Versions

**Goal:** Register several model versions with different algorithms.

```python
# exercise2_multiple_versions.py
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("05-model-registry")

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

MODEL_NAME = "wine-quality-classifier"

models = {
    "RandomForest-200trees": RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
}

for name, model in models.items():
    with mlflow.start_run(run_name=f"register-{name}"):
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", accuracy)

        signature = infer_signature(X_test_s, y_pred)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

        print(f"{name}: accuracy={accuracy:.4f}")

# List all versions
client = mlflow.tracking.MlflowClient()
for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
    print(f"  Version {mv.version} | Stage: {mv.current_stage} | Run: {mv.run_id[:8]}")
```

### Exercise 3: Stage Transitions

**Goal:** Move models through the lifecycle stages programmatically.

```python
# exercise3_stage_transitions.py
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

MODEL_NAME = "wine-quality-classifier"

# Find the best version by accuracy
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
best_version = None
best_accuracy = 0

for mv in versions:
    run = client.get_run(mv.run_id)
    accuracy = run.data.metrics.get("accuracy", 0)
    print(f"  Version {mv.version} | Stage: {mv.current_stage} | Accuracy: {accuracy:.4f}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_version = mv.version

print(f"\nBest version: {best_version} (accuracy: {best_accuracy:.4f})")

# Promote to Staging
client.transition_model_version_stage(
    name=MODEL_NAME, version=best_version, stage="Staging",
    archive_existing_versions=False,
)
print(f"Version {best_version} -> Staging")

# Add a description
client.update_model_version(
    name=MODEL_NAME, version=best_version,
    description=f"Best accuracy ({best_accuracy:.4f}) as of {datetime.now().strftime('%Y-%m-%d')}.",
)

# Promote to Production (archive existing Production versions)
client.transition_model_version_stage(
    name=MODEL_NAME, version=best_version, stage="Production",
    archive_existing_versions=True,
)
print(f"Version {best_version} -> Production")
```

### Exercise 4: Automated Promotion with Validation

**Goal:** Build a promotion script that validates models before promoting.

```python
# exercise4_auto_promote.py
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

MODEL_NAME = "wine-quality-classifier"
ACCURACY_THRESHOLD = 0.90
F1_THRESHOLD = 0.88


def validate_model(model_name: str, version: int) -> dict:
    """Run validation checks against a held-out set."""
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.pyfunc.load_model(model_uri)

    wine = load_wine()
    _, X_val, _, y_val = train_test_split(
        wine.data, wine.target, test_size=0.3, random_state=99
    )

    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    f1 = f1_score(y_val, predictions, average="weighted")

    results = {
        "accuracy": accuracy,
        "f1_score": f1,
        "passes_accuracy": accuracy >= ACCURACY_THRESHOLD,
        "passes_f1": f1 >= F1_THRESHOLD,
    }

    print(f"  Accuracy:  {accuracy:.4f} ({'PASS' if results['passes_accuracy'] else 'FAIL'})")
    print(f"  F1 Score:  {f1:.4f} ({'PASS' if results['passes_f1'] else 'FAIL'})")
    return results


def auto_promote(model_name: str):
    versions = client.search_model_versions(f"name='{model_name}'")
    candidates = [mv for mv in versions if mv.current_stage == "None"]

    if not candidates:
        print("No candidate versions to promote.")
        return

    latest = max(candidates, key=lambda mv: int(mv.version))
    print(f"Validating version {latest.version}...")
    results = validate_model(model_name, int(latest.version))

    if results["passes_accuracy"] and results["passes_f1"]:
        client.transition_model_version_stage(
            name=model_name, version=latest.version, stage="Staging",
        )
        for key, value in results.items():
            client.set_model_version_tag(
                name=model_name, version=latest.version,
                key=f"validation_{key}", value=str(value),
            )
        print(f"Version {latest.version} promoted to Staging!")
    else:
        print(f"Version {latest.version} failed validation.")


auto_promote(MODEL_NAME)
```

### Exercise 5: Load Production Models for Serving

**Goal:** Load models by stage for deployment.

```python
# exercise5_load_by_stage.py
import mlflow
from sklearn.datasets import load_wine

mlflow.set_tracking_uri("http://localhost:5000")

MODEL_NAME = "wine-quality-classifier"

# Load by stage name
production_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
print(f"Production model loaded: {type(production_model)}")

wine = load_wine()
sample = wine.data[:5]
predictions = production_model.predict(sample)
print(f"Predictions: {predictions}")
print(f"Class names: {[wine.target_names[p] for p in predictions]}")

# Load by specific version number
v1_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/1")
print(f"\nVersion 1 model loaded: {type(v1_model)}")
```

---

## Using the Model Registry CLI

The provided `src/registry/model_registry.py` script wraps these operations:

```bash
# Register a model from a run
python -m src.registry.model_registry register --run-id <RUN_ID> --model-name "wine-classifier"

# Promote to Staging
python -m src.registry.model_registry promote --model-name "wine-classifier" --version 2 --stage Staging

# Promote to Production
python -m src.registry.model_registry promote --model-name "wine-classifier" --version 2 --stage Production

# Compare versions
python -m src.registry.model_registry compare --model-name "wine-classifier" --versions 1,2,3

# List all registered models
python -m src.registry.model_registry list
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Using file-based tracking | `Model registry not available` | Use database-backed tracking server |
| Forgetting to archive old Production | Two versions in Production | Use `archive_existing_versions=True` |
| Hardcoding version numbers | Breaks when new versions added | Use stage names or `latest_versions()` |
| Registering without signature | No input validation | Always infer and log the signature |

---

## Self-Check Questions

1. What are the four stages in the Model Registry lifecycle?
2. Why does the Registry require a database-backed tracking server?
3. How do you load the current Production model?
4. What happens to the old Production version when you promote a new one?
5. How would you build CI/CD that auto-validates and promotes models?

---

## You Know You Have Completed This Module When...

- [ ] You registered a model from a training run
- [ ] You created multiple versions of the same registered model
- [ ] You promoted a model through None -> Staging -> Production
- [ ] You built an automated validation and promotion script
- [ ] You loaded a model by stage name for inference
- [ ] Validation script passes: `bash modules/05-model-packaging/validation/validate.sh`

---

**Next: [Module 06 - Model Serving -->](../06-model-serving/)**
