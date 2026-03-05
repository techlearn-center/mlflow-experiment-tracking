# Module 03: Model Logging and Signatures - sklearn, PyTorch, and Custom Models

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 02 completed, familiarity with sklearn and PyTorch basics |

---

## Learning Objectives

By the end of this module, you will be able to:

- Log sklearn models with `mlflow.sklearn.log_model()` and inferred signatures
- Log PyTorch models with `mlflow.pytorch.log_model()`
- Create custom MLflow model flavors using `mlflow.pyfunc`
- Understand model signatures and input examples
- Load logged models for inference using different flavors
- Package preprocessing steps with your model

---

## Concepts

### What is a Model Signature?

A model signature defines the expected input and output schema of your model. It acts as a contract that is enforced at inference time.

```python
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# A signature specifying column names and types
input_schema = Schema([
    ColSpec("double", "sepal_length"),
    ColSpec("double", "sepal_width"),
    ColSpec("double", "petal_length"),
    ColSpec("double", "petal_width"),
])
output_schema = Schema([ColSpec("long", "predicted_class")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
```

In practice, you can **infer** signatures automatically:

```python
from mlflow.models.signature import infer_signature
signature = infer_signature(X_test, y_pred)
```

### MLflow Model Flavors

MLflow uses "flavors" to support different ML frameworks:

| Flavor | Import | Use Case |
|---|---|---|
| `mlflow.sklearn` | scikit-learn models | Classification, regression, clustering |
| `mlflow.pytorch` | PyTorch models | Deep learning, NLP, computer vision |
| `mlflow.xgboost` | XGBoost models | Gradient boosting |
| `mlflow.tensorflow` | TensorFlow/Keras models | Deep learning |
| `mlflow.pyfunc` | Any Python function | Custom models, ensembles, pipelines |

Every logged model also gets a `pyfunc` flavor, meaning you can load **any** model with `mlflow.pyfunc.load_model()`.

### Input Examples

An input example is a sample input saved alongside the model. It documents what the model expects and can be used for testing:

```python
mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    signature=signature,
    input_example=X_test.iloc[:3],  # First 3 rows as example
)
```

---

## Hands-On Lab

### Exercise 1: Log an sklearn Model with Signature

**Goal:** Train and log a scikit-learn pipeline with automatic signature inference.

```python
# exercise1_sklearn_model.py
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("03-sklearn-models")

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="class")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="sklearn-pipeline"):
    # Build a pipeline with preprocessing + model
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = float((y_test == y_pred).mean())

    mlflow.log_param("model_type", "Pipeline(StandardScaler + RandomForest)")
    mlflow.log_metric("accuracy", accuracy)

    # Infer signature from actual data
    signature = infer_signature(X_test, y_pred)
    print(f"Signature:\n{signature}")

    # Log the entire pipeline as a model
    model_info = mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_test.iloc[:3],
        registered_model_name="wine-pipeline",
    )

    print(f"\nModel URI: {model_info.model_uri}")
    print(f"Accuracy: {accuracy:.4f}")

# ---- Load and test the model ----
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
test_predictions = loaded_model.predict(X_test.iloc[:5])
print(f"\nLoaded model predictions: {test_predictions}")
```

### Exercise 2: Log a PyTorch Model

**Goal:** Train a simple PyTorch neural network and log it with MLflow.

```python
# exercise2_pytorch_model.py
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("03-pytorch-models")


class IrisClassifier(nn.Module):
    """Simple feedforward network for Iris classification."""

    def __init__(self, input_dim=4, hidden_dim=32, output_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def train_pytorch_model():
    iris = load_iris()
    X, y = iris.data, iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    with mlflow.start_run(run_name="pytorch-iris-classifier"):
        hidden_dim = 32
        lr = 0.01
        epochs = 100

        mlflow.log_params({
            "model_type": "PyTorch_FeedForward",
            "hidden_dim": hidden_dim,
            "learning_rate": lr,
            "epochs": epochs,
            "optimizer": "Adam",
            "loss_fn": "CrossEntropyLoss",
        })

        model = IrisClassifier(input_dim=4, hidden_dim=hidden_dim, output_dim=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_t)
                    _, predicted = torch.max(test_outputs, 1)
                    test_acc = (predicted == y_test_t).float().mean().item()
                    train_outputs = model(X_train_t)
                    _, train_pred = torch.max(train_outputs, 1)
                    train_acc = (train_pred == y_train_t).float().mean().item()

                mlflow.log_metrics({
                    "train_loss": loss.item(),
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                }, step=epoch + 1)
                model.train()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            _, predicted = torch.max(test_outputs, 1)
            final_accuracy = (predicted == y_test_t).float().mean().item()

        mlflow.log_metric("final_test_accuracy", final_accuracy)

        sample_input = X_test[:3].astype(np.float32)
        sample_output = predicted[:3].numpy()
        signature = infer_signature(sample_input, sample_output)

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=sample_input,
        )

        print(f"Final Test Accuracy: {final_accuracy:.4f}")


train_pytorch_model()
```

### Exercise 3: Custom Model with mlflow.pyfunc

**Goal:** Create a custom model class that wraps preprocessing + inference + postprocessing.

```python
# exercise3_custom_pyfunc.py
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("03-custom-pyfunc")


class WineClassifierWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow model that bundles:
    - Feature scaling (StandardScaler)
    - Classification (RandomForest)
    - Output mapping (numeric class -> class name)
    """

    def __init__(self, scaler, model, class_names):
        self.scaler = scaler
        self.model = model
        self.class_names = class_names

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            features = model_input.values
        else:
            features = np.array(model_input)

        scaled = self.scaler.transform(features)
        predictions = self.model.predict(scaled)
        named_predictions = [self.class_names[p] for p in predictions]

        return named_predictions


def train_custom_model():
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target
    class_names = list(wine.target_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    wrapper = WineClassifierWrapper(scaler, model, class_names)

    with mlflow.start_run(run_name="custom-pyfunc-model"):
        mlflow.log_params({
            "model_type": "custom_pyfunc",
            "inner_model": "RandomForest",
            "preprocessing": "StandardScaler",
            "output_format": "class_names",
        })

        sample_predictions = wrapper.predict(context=None, model_input=X_test.iloc[:5])
        print(f"Sample predictions: {sample_predictions}")

        signature = infer_signature(X_test, sample_predictions)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapper,
            signature=signature,
            input_example=X_test.iloc[:3],
        )

        y_pred = model.predict(scaler.transform(X_test))
        mlflow.log_metric("accuracy", float(np.mean(y_test == y_pred)))

    # Verify the loaded model works
    run_id = mlflow.last_active_run().info.run_id
    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    result = loaded.predict(X_test.iloc[:5])
    print(f"\nLoaded model predictions: {result}")


train_custom_model()
```

### Exercise 4: Loading Models with Different Flavors

**Goal:** Understand the difference between loading with native flavor vs pyfunc.

```python
# exercise4_model_flavors.py
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("03-model-flavors")

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="flavor-comparison") as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")

run_id = run.info.run_id
model_uri = f"runs:/{run_id}/model"

# Load as pyfunc -- generic interface, only .predict()
pyfunc_model = mlflow.pyfunc.load_model(model_uri)
pyfunc_preds = pyfunc_model.predict(X_test.iloc[:5])
print(f"pyfunc predictions: {pyfunc_preds}")
print(f"pyfunc model type: {type(pyfunc_model)}")

# Load as native sklearn -- full sklearn API available
sklearn_model = mlflow.sklearn.load_model(model_uri)
sklearn_preds = sklearn_model.predict(X_test.iloc[:5])
sklearn_proba = sklearn_model.predict_proba(X_test.iloc[:5])
print(f"\nsklearn predictions: {sklearn_preds}")
print(f"sklearn probabilities:\n{sklearn_proba}")
print(f"sklearn model type: {type(sklearn_model)}")
print(f"Number of estimators: {sklearn_model.n_estimators}")

# Key difference:
# - pyfunc: only .predict() available -- best for serving and deployment
# - native: full API (.predict_proba, .feature_importances_, etc.) -- best for analysis
```

---

## Model Artifacts Explained

When you log a model, MLflow creates this directory structure:

```
model/
  ├── MLmodel              # Metadata: flavors, signature, requirements
  ├── model.pkl            # Serialized model (sklearn)
  ├── conda.yaml           # Conda environment specification
  ├── python_env.yaml      # Python virtualenv specification
  ├── requirements.txt     # Pip requirements
  └── input_example.json   # Sample input data
```

The `MLmodel` file describes which flavors are available:

```yaml
flavors:
  python_function:
    env: conda.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    python_version: 3.11.0
  sklearn:
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.4.0
signature:
  inputs: '[{"name": "sepal_length", "type": "double"}, ...]'
  outputs: '[{"type": "long"}]'
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Forgetting to infer signature | Model loads but no input validation | Always call `infer_signature()` |
| Logging model outside a run | `MlflowException: No active run` | Wrap in `with mlflow.start_run():` |
| Wrong input shape on predict | Shape mismatch error | Check signature and pass matching DataFrame |
| Using pickle instead of MLflow | No versioning, no signature, no portability | Use `mlflow.<flavor>.log_model()` |

---

## Self-Check Questions

1. What is the difference between `mlflow.sklearn.load_model()` and `mlflow.pyfunc.load_model()`?
2. Why should you always include a model signature?
3. When would you create a custom `mlflow.pyfunc.PythonModel`?
4. What files does MLflow create in the model artifact directory?
5. How does `input_example` help during model deployment?

---

## You Know You Have Completed This Module When...

- [ ] You logged an sklearn pipeline with inferred signature
- [ ] You logged a PyTorch model with step-based training metrics
- [ ] You created and logged a custom pyfunc model
- [ ] You loaded models using both native and pyfunc flavors
- [ ] You can explain what the MLmodel file contains
- [ ] Validation script passes: `bash modules/03-experiment-tracking/validation/validate.sh`

---

**Next: [Module 04 - Model Registry and Versioning -->](../04-model-registry/)**
